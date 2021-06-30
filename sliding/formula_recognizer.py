from PIL import ImageGrab, Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn import preprocessing


class Recognizer(object):

    def __init__(self, model_path):
        self.labels = ['decimal', 'div', 'eight', 'equal', 'five', 'four', 'left', 'minus', 'nine', 'one', 'plus',
                       'right', 'seven',
                       'six', 'three',
                       'times', 'two', 'zero']
        self.mode_path = model_path
        self.model_shape = (1, 100, 100, 1)
        self.model_image_size = (100, 100)
        self.minimum_score = 0.1
        self.expression_margin_threshold = 2  # 2 times larger than average margin
        self.blank_margin_ratio = 1 / 5  # keep 1/5 height as margin to improve score
        self.model = load_model(model_path)

    def slice_horizontally(self, image):
        """
        check how many rows of formulas does the image have
        :param image: black background image
        :return: the sliced horizontal parts of the image
        """
        ret = []  # (actual sliced images...)
        slices = []  # sliced coordinates (y1, y2)
        on_row = False
        pre_y = 0
        for i in range(image.shape[0]):
            row = image[i, :]
            row_sum = np.sum(row)
            if row_sum == 0:
                if on_row:
                    slices.append((pre_y, i))
                    on_row = False
                else:
                    pre_y = i
            else:
                if on_row:
                    continue
                else:
                    on_row = True
                    continue
        for slice in slices:
            y1, y2 = slice
            sliced_img = image[y1: y2, :]
            ret.append((sliced_img, y1))  # also stores original coordinates
        return ret

    def get_outer_frame(self, image):
        """
        get the outer frame of the written formula
        :param image: numpy array representing the image with black background
        :return: the coordinates of the frame (x1, y1, x2, y2)
        """
        non_zero = np.nonzero(image)
        x, y = non_zero  # here x is actually axis=0, which is actually y by coordinates
        x1 = np.min(y)
        y1 = np.min(x)
        x2 = np.max(y)
        y2 = np.max(x)
        return x1, y1, x2, y2

    def invert_image(self, image):
        """
        invert a white background image to black ground image
        :param image:
        :return:
        """
        return 255 - image

    def transfer_to_model_shape(self, image):
        ret = image.reshape(self.model_shape)
        return ret

    def pred_to_label(self, pred):
        arg_sorted = np.argsort(pred)
        max_idx = arg_sorted[0][-1]
        final_label = None
        score = 0.0
        if pred[0][max_idx] > self.minimum_score:
            final_label, score = self.labels[max_idx], pred[0][max_idx]
        return final_label, score

    def segment_symbols_one_row(self, image, outer_frame):
        """
        :param image: image with black background
        :param outer_frame:
        :return:
        """
        ret = []
        x1, y1, x2, y2 = outer_frame
        pre_idx = x1
        on_symbol = True
        expressions = []
        margins = []
        symbol_count = 0
        for i in range(x1 + 1, x2 + 1):  # TODO: add boundary check
            bar = image[y1: y2 + 1, i - 1: i]  # TODO: add boundary check
            sum = np.sum(bar)
            if sum == 0:
                # nothing in the bar
                if on_symbol:
                    # found one
                    s_x1 = pre_idx
                    s_y1 = y1
                    s_x2 = i
                    s_y2 = y2
                    pre_idx = i
                    ret.append((s_x1, s_y1, s_x2, s_y2))
                    on_symbol = False

                    if len(ret) == 1:
                        # first one, no margin, so skip
                        pass
                    else:
                        current_x1 = ret[symbol_count][0]
                        pre_x2 = ret[symbol_count - 1][2]
                        margin = current_x1 - pre_x2
                        margins.append(margin)

                    symbol_count += 1
                else:
                    pre_idx = i
                    continue
            else:
                if on_symbol:
                    continue
                else:
                    on_symbol = True
                    pre_idx = i

        if on_symbol:
            # last one
            s_x1 = pre_idx
            s_y1 = y1
            s_x2 = x2 + 1  # TODO: boundary check?
            s_y2 = y2
            ret.append((s_x1, s_y1, s_x2, s_y2))

        # after getting all symbols in one row, then we can check the possible expression split
        margin_mean = np.mean(margins)
        idx = 0
        for margin in margins:
            if margin / margin_mean > self.expression_margin_threshold:
                # new expression split found
                expressions.append(idx)
            idx += 1

        # the last one is also an expression split
        expressions.append(symbol_count)
        return ret, np.array(expressions)

    def slice_scale_image(self, rec, image, size):
        """
        image as black background
        slice the image and then make it as target size
        :param rec:
        :param image:
        :param size:
        :return:
        """
        x1, y1, x2, y2 = rec
        sliced = image[y1: y2 + 1, x1: x2 + 1]  # TODO: check the boundary
        # centroid = get_centroid(sliced)
        # c_x, c_y = centroid
        # put it center to make the model predict in higher score
        # add 1/5 on the top and bottom
        blank_height = int((y2 - y1) * self.blank_margin_ratio)
        top_bottom_blank = np.zeros((blank_height, (x2 - x1 + 1)))
        top_added = np.concatenate([top_bottom_blank, sliced], axis=0)
        bottom_added = np.concatenate([top_added, top_bottom_blank], axis=0)

        # make rectangle to square so that the resize will not be distorted
        width = x2 - x1 + 1
        height = y2 - y1 + 1 + 2 * blank_height
        if height > width:
            supplement = int((height - width) / 2)
            s_image = np.zeros((height, supplement))
            image_left = np.concatenate([s_image, bottom_added], axis=1)
            image_supplemented = np.concatenate([image_left, s_image], axis=1)
        else:
            supplement = int((width - height) / 2)
            s_image = np.zeros((supplement, width))
            image_up = np.concatenate([s_image, bottom_added], axis=0)
            image_supplemented = np.concatenate([image_up, s_image], axis=0)

        # ret = transform.resize(image_supplemented, size)
        ret = cv2.resize(image_supplemented, dsize=self.model_image_size, interpolation=cv2.INTER_NEAREST)
        return ret

    def shrink_box(self, image, rec):
        """
        assume black background
        :param image:
        :param rec:
        :return: return the shrunk rectangle
        """
        x1, y1, x2, y2 = rec
        idx = y1
        # shrink from above
        for i in range(y1, y2):
            row = image[i, x1: x2]
            row_sum = np.sum(row)
            if row_sum == 0:
                idx = i
            else:
                break
        new_y1 = idx
        idx = y2
        for i in reversed(range(y1, y2 + 1)):
            row = image[i, x1: x2]
            row_sum = np.sum(row)
            if row_sum == 0:
                idx = i
            else:
                break
        new_y2 = idx
        # double check if decimal then keep the box, other it will be distorted when scaling
        if (new_y2 - new_y1) < (y2 - y1) / 4:
            return x1, y1, x2, y2
        else:
            return x1, new_y1, x2, new_y2

    def pil_image_to_numpy(self, pil_image):
        """
        :param pil_image: the PIL Image object with white background
        :return: numpy array represent image with black background
        """
        img = pil_image.convert('L')
        # img = ImageOps.invert(img)
        pix = np.array(img)
        # test_image = pix.astype(np.float32) / 255
        image = pix.astype(np.float32) / 1
        image = self.invert_image(image)
        return image

    def recognize(self, image):
        """
        :param image: must be black background
        :return: (list(x1, y1, x2, y2, label, score), list(expression index))
        """
        hori_sliced_images = self.slice_horizontally(image)
        full_segments = []
        full_expressions = []
        expression_idx = 0
        for sliced_image, origin_y in hori_sliced_images:
            frame = self.get_outer_frame(sliced_image)
            (x1, y1, x2, y2) = frame
            frame_on_origin = (x1, y1 + origin_y, x2, y2 + origin_y)
            segments, expressions = self.segment_symbols_one_row(image, frame_on_origin)
            full_segments.extend(segments)

            global_idx = expressions + expression_idx
            full_expressions.extend(global_idx)
            expression_idx += expressions[-1] + 1

        shrunk_segments = []
        for (x1, y1, x2, y2) in full_segments:
            new_rec = self.shrink_box(image, (x1, y1, x2, y2))
            shrunk_segments.append(new_rec)

        # for (x1, y1, x2, y2) in shrunk_segments:
        #     cv2.rectangle(test_image_before_nms, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        ret_symbols = []
        for (x1, y1, x2, y2) in shrunk_segments:
            sliced_image = self.slice_scale_image((x1, y1, x2, y2), image, (100, 100))
            # as the model we trained is white background
            white_back = self.invert_image(sliced_image)
            model_shaped = self.transfer_to_model_shape(white_back)
            predicted = self.model.predict(model_shaped)
            label, score = self.pred_to_label(predicted)
            ret_symbols.append((x1, y1, x2, y2, label, score))
        return ret_symbols, full_expressions
