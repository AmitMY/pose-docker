import math
import os

import cv2
import numpy as np


def read_image(image):
    image = cv2.imread(image)
    size = max(max(image.shape), 368)

    scale = size / (image.shape[0] * 1.0)
    im = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((size, size, 3)) * 128

    img_h = im.shape[0]
    img_w = im.shape[1]
    if img_w < size:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(size / 2 - math.floor(img_w / 2)):int(size / 2 + math.floor(img_w / 2) + offset), :] = im
    else:
        # crop the center of the origin image
        output_img = im[:, int(img_w / 2 - size / 2):int(img_w / 2 + size / 2), :]
    return output_img


if __name__ == "__main__":
    files = os.listdir("hands")

    for f in files:
        im = read_image('hands/' + f)
        cv2.imwrite('new_hands/' + f, im)
