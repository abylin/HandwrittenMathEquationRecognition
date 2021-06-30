import formula_recognizer
from PIL import ImageGrab, Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn import preprocessing

recognizer = formula_recognizer.Recognizer(".//..//mix.h5")


img = Image.open(".//mywrite3.png")
img = img.convert('L')
# img = ImageOps.invert(img)
pix = np.array(img)
# test_image = pix.astype(np.float32) / 255
test_image = pix.astype(np.float32) / 1
test_image = recognizer.invert_image(test_image)
#
# plt.imshow(test_image)
# plt.title('Original image')
# plt.xticks([]), plt.yticks([])
# # plt.show()
#
# symbols, expression_idx = recognizer.recognize(test_image)
# for symb in symbols:
#     print(symb)
# print(expression_idx)
#
# for (x1, y1, x2, y2, label, score) in symbols:
#     cv2.rectangle(test_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
#
# plt.title('Detected characters')
# plt.imshow(test_image)
# plt.xticks([]), plt.yticks([])
# plt.show()

# show the histogram horizontally
image_copy = np.copy(test_image)
horizontal_sum = np.sum(image_copy, axis=1, keepdims=True)
plot_y = horizontal_sum.flatten()
plot_x = range(0, horizontal_sum.shape[0])
plt.title('horizontal histogram')
plt.bar(plot_x, plot_y)
plt.show()

# show the histogram vertically by each line
image_copy = np.copy(test_image)
up_image = image_copy[:150, :]
down_image = image_copy[150:, :]
vertical_sum_up = np.sum(up_image, axis=0, keepdims=True)
plot_y_up = vertical_sum_up.flatten()
vertical_sum_down = np.sum(down_image, axis=0, keepdims=True)
plot_y_down = vertical_sum_down.flatten()
plot_x = range(0, plot_y_up.shape[0])
fig, axs = plt.subplots(2)
fig.suptitle('Vertically histogram')
axs[0].plot(plot_x, plot_y_up)
axs[1].plot(plot_x, plot_y_down)
plt.show()
