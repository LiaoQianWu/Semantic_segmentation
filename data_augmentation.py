import os
import cv2
import numpy as np


def horizontal_shift(im, ratio):
    if ratio > 1 or ratio < 0:
        print("Value should be less than 1 and greater than 0.")
        return im
    ratio = np.random.uniform(-ratio, ratio)
    height, width, channel = im.shape
    to_shift = int(width*ratio)
    if ratio > 0:
        im = im[:, :(width-to_shift)]
    if ratio < 0:
        im = im[:, (-to_shift):]
    im = cv2.resize(im, (height, width), cv2.INTER_CUBIC)
    return im        


def horizontal_flip(im, ma, flag):
    if flag:
        return cv2.flip(im, 1), cv2.flip(ma, 1)
    else:
        return im, ma

    
def vertical_flip(im, ma, flag):
    if flag:
        return cv2.flip(im, 0), cv2.flip(ma, 0)
    else:
        return im, ma


def rotation(im, ma, angle):
    angle = int(np.random.uniform(-angle, angle))
    height, width, channel = im.shape
    M = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)
    im = cv2.warpAffine(im, M, (width, height))
    ma = cv2.warpAffine(ma, M, (width, height))
    return im, ma


# Other data augmentation functions
"""
def brightness(im, lower_bound, upper_bound):
    value = np.random.uniform(lower_bound, upper_bound)
    im = im * value # Can be either addition or multiplication
    im[im > (2**16 -1)] = 2**16 - 1
    #im = np.clip(im, 0, 2**16-1)
    return im
"""
