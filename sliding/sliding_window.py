from skimage import transform
from PIL import ImageGrab, Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit_transform(
    ['decimal', 'div', 'eight', 'equal', 'five', 'four', 'minus', 'nine', 'one', 'plus', 'seven', 'six', 'three',
     'times', 'two', 'zero'])

labels = ['decimal', 'div', 'eight', 'equal', 'five', 'four', 'minus', 'nine', 'one', 'plus', 'seven', 'six', 'three',
          'times', 'two', 'zero']

output_folder_name = ".\\output"
test_image_path = ".\\mywrite2.png"
dims = (100, 100)
n_channels = 1
step_size = 20  # 16
window_size = dims
min_size = (100, 100)  # 100, 100
downscale = 1.4
# NMS threshold
threshold = 0.000001


def pyramid(image, downscale=1.5, min_size=(64, 64)):
    yield image

    while True:
        w = int(image.shape[1] / downscale)
        image = resize(image, width=w)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def non_max_suppression(boxes, class_scores, overlap_thresh=0.7):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # TODO: add process for class index 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int"), class_scores[pick]


def is_all_black(img):
    normed = img / 255
    mean = normed.mean()
    if np.abs(mean - 1.0) <= 0.0000001:
        return True
    else:
        return False


# assume grey scale image with white background
def get_centroid(image):
    image = 255 - image
    m = cv2.moments(image)
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return x, y


def is_centroid_in_window(x, y, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x1 < x < x2) and (y1 < y < y2)


def slice_horizontally(image):
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


def get_outer_frame(image):
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


def invert_image(image):
    """
    invert a white background image to black ground image
    :param image:
    :return:
    """
    return 255 - image


def transfer_to_model_shape(image):
    ret = image.reshape((1, 100, 100, 1))
    return ret


def pred_to_label(pred):
    arg_sorted = np.argsort(pred)
    max_idx = arg_sorted[0][-1]
    final_label = None
    score = 0.0
    if pred[0][max_idx] > 0.5:
        final_label, score = labels[max_idx], pred[0][max_idx]
    return final_label, score


def segment_symbols_one_row(image, outer_frame):
    """
    :param image: image with black background
    :param outer_frame:
    :return:
    """
    ret = []
    x1, y1, x2, y2 = outer_frame
    bar_height = y2 - y1
    pre_idx = x1
    on_symbol = True
    expressions = []
    width_sum, symbol_count, width_average = 0, 0, 0
    gap_threshold = 2  # 2 times larger than average symbol width
    for i in range(x1 + 1, x2 + 1):  # TODO: add boundary check
        bar = image[y1: y2+1, i-1: i]  # TODO: add boundary check
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

                symbol_count += 1
                width_sum += (s_x2 - s_x1 + 1)
                width_average = width_sum / symbol_count
            else:
                pre_idx = i
                continue
        else:
            if on_symbol:
                continue
            else:
                on_symbol = True
                pre_idx = i

                current_gap = i - ret[-1][2]
                if current_gap / width_average > gap_threshold:
                    expressions.append(symbol_count - 1)  # record the last symbol index as an expression

    if on_symbol:
        # last one
        s_x1 = pre_idx
        s_y1 = y1
        s_x2 = x2 + 1  # TODO: boundary check?
        s_y2 = y2
        ret.append((s_x1, s_y1, s_x2, s_y2))
        expressions.append(symbol_count)

    return ret, expressions


def slice_scale_image(rec, image, size):
    """
    image as black background
    slice the image and then make it as target size
    :param rec:
    :param image:
    :param size:
    :return:
    """
    x1, y1, x2, y2 = rec
    sliced = image[y1: y2+1, x1: x2+1]  # TODO: check the boundary
    # centroid = get_centroid(sliced)
    # c_x, c_y = centroid
    # put it center to make the model predict in higher score
    # add 1/5 on the top and bottom
    blank_height = int((y2 - y1)/5)
    top_bottom_blank = np.zeros((blank_height, (x2 - x1 + 1)))
    top_added = np.concatenate([top_bottom_blank, sliced], axis=0)
    bottom_added = np.concatenate([top_added, top_bottom_blank], axis=0)

    # make rectangle to square so that the resize will not be distorted
    width = x2 - x1 + 1
    height = y2 - y1 + 1 + 2 * blank_height
    if height > width:
        supplement = int((height - width)/2)
        s_image = np.zeros((height, supplement))
        image_left = np.concatenate([s_image, bottom_added], axis=1)
        image_supplemented = np.concatenate([image_left, s_image], axis=1)
    else:
        supplement = int((width - height)/2)
        s_image = np.zeros((supplement, width))
        image_up = np.concatenate([s_image, bottom_added], axis=0)
        image_supplemented = np.concatenate([image_up, s_image], axis=0)

    # ret = transform.resize(image_supplemented, size)
    ret = cv2.resize(image_supplemented, dsize=size, interpolation=cv2.INTER_NEAREST)
    return ret


def shrink_box(image, rec):
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
    if (new_y2 - new_y1) < (y2 - y1)/4:
        return x1, y1, x2, y2
    else:
        return x1, new_y1, x2, new_y2




def main():

    model = load_model(os.path.join(".//../", 'mix.h5'))

    img = Image.open(test_image_path)
    img = img.convert('L')
    # img = ImageOps.invert(img)
    pix = np.array(img)
    # test_image = pix.astype(np.float32) / 255
    test_image = pix.astype(np.float32) / 1
    test_image = invert_image(test_image)

    path, filename = os.path.split(test_image_path)
    filename = os.path.splitext(filename)[0]
    test_image_before_nms_path = os.path.join(path, filename + '_before_nms.png')
    test_image_after_nms_path = os.path.join(path, filename + '_after_nms.png')

    plt.imshow(test_image)
    plt.title('Original image')
    plt.xticks([]), plt.yticks([])
    plt.show()

    detections = []
    detections_c = []
    # downscale
    # downscale_power = 0
    test_image_clone = test_image.copy()
    # for scaled_image in pyramid(test_image, downscale, min_size):
    #     for (x, y, window) in sliding_window(scaled_image, step_size, window_size):
    #         if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
    #             continue
    #         # print("window {}, {} at {} {} on scaled image size{}".format(window_size[0], window_size[1], x, y, scaled_image.shape))
    #         # reshaped = window.reshape(1, *dims, n_channels)
    #         if is_all_black(window):
    #             continue
    #         wind = np.expand_dims(window, axis=0)
    #         reshaped = wind.reshape(*dims, 1)
    #         expanded = np.expand_dims(reshaped, axis=0)
    #         predicted = model.predict(expanded)
    #
    #         # predicted = predicted.reshape(-1)
    #
    #         label, score = ret_to_label(predicted)
    #         if label is not None:
    #
    #             x1 = int(x * (downscale ** downscale_power))
    #             y1 = int(y * (downscale ** downscale_power))
    #
    #             detections.append((x1, y1,
    #                                x1 + int(window_size[0] * (downscale ** downscale_power)),
    #                                y1 + int(window_size[1] * (downscale ** downscale_power))))
    #             detections_c.append((label, score))
    #
    #     # downscale
    #     downscale_power += 1

    test_image_before_nms = test_image_clone.copy()
    hori_sliced_images = slice_horizontally(test_image_before_nms)
    full_segments = []
    for sliced_image, origin_y in hori_sliced_images:
        frame = get_outer_frame(sliced_image)
        (x1, y1, x2, y2) = frame
        frame_on_origin = (x1, y1 + origin_y, x2, y2 + origin_y)
        segments, expressions = segment_symbols_one_row(test_image_before_nms, frame_on_origin)
        full_segments.extend(segments)

    shrunk_segments = []
    for (x1, y1, x2, y2) in full_segments:
        new_rec = shrink_box(test_image, (x1, y1, x2, y2))
        shrunk_segments.append(new_rec)

    for (x1, y1, x2, y2) in shrunk_segments:
        cv2.rectangle(test_image_before_nms, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
    # cv2.rectangle(test_image_before_nms, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
    # for (x1, y1, x2, y2) in detections:  # TODO: process class label
    #     cv2.rectangle(test_image_before_nms, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
    #
    plt.title('Detected cars befor NMS')
    plt.imshow(test_image_before_nms)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imsave(test_image_before_nms_path, test_image_before_nms)

    #
    def math_symbol_suppression(boxes, classes_score, overlap_thresh=0.7):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("int")

        pick, ret = [], []
        prev_c = ""
        prev_i = -1
        prev_s = 0.0
        # reduce and only leave nearby max score
        for i in range(len(classes_score)):
            c, s = classes_score[i]
            if c != prev_c:  # anyway need record
                if prev_i == -1:  # first
                    prev_c, prev_s, prev_i = c, s, i
                else:  # new class starts, append old class highest
                    pick.append(prev_i)
                    prev_c, prev_s, prev_i = c, s, i
            else:  # the same class continues
                if prev_s < s:  # prev score is smaller, use current
                    prev_c, prev_s, prev_i = c, s, i
                else:
                    pass  # still use old
        pick.append(prev_i)

        # reduce overlap by centroid
        def is_in_ret_boxes(ret_idx, boxes, centroid, c_s):
            if not ret_idx:
                return False
            x, y = centroid
            for i in ret_idx:
                in_box = is_centroid_in_window(x, y, boxes[i])
                if in_box:
                    return True
                else:
                    continue
            return False

        for i in pick:
            box = boxes[i]
            x1, y1, x2, y2 = box
            box_image = test_image_clone[y1: y2, x1: x2]
            center = get_centroid(box_image)
            x = center[0] + x1
            y = center[1] + y1
            in_ret = is_in_ret_boxes(ret, boxes, (x, y), classes_score[i])
            if in_ret:
                continue  # abandon it as it overlaps with previous one
            else:
                ret.append(i)

        return boxes[ret].astype("int"), classes_score[ret]

    # # Non-Maxima Suppression
    # detections_nms, detections_c_nms = math_symbol_suppression(np.array(detections), np.array(detections_c), threshold)
    # print("detections after nms " + str(detections_c_nms))
    # test_image_after_nms = test_image_clone
    # for (x1, y1, x2, y2) in detections_nms:
    #     cv2.rectangle(test_image_after_nms, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
    #
    # plt.title('Detected cars after NMS')
    # plt.imshow(test_image_after_nms)
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    # plt.imsave(test_image_after_nms_path, test_image_after_nms)
    # pass

    test_image = test_image_clone.copy()
    for (x1, y1, x2, y2) in shrunk_segments:
        sliced_image = slice_scale_image((x1, y1, x2, y2), test_image, (100, 100))
        # as the model in trained in white background
        white_back = invert_image(sliced_image)
        model_shaped = transfer_to_model_shape(white_back)
        predicted = model.predict(model_shaped)
        label, score = pred_to_label(predicted)
        print("label {}, score {}".format(label, score))

    test_image = test_image_clone.copy()
    print("expression as " + str(expressions))

if __name__ == "__main__":
    main()
