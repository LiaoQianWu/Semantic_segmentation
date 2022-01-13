import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def brightness(im, lower_bound, upper_bound):
    value = np.random.uniform(lower_bound, upper_bound)
    im = im * value #can either addition or multiplication
    im[im > (2**16 -1)] = 2**16 - 1
    #im = np.clip(im, 0, 2**16-1)
    return im


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


"""
#TEST
im = cv2.imread("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/data/train_frames/train/bordered_cropped_C1-20190704_CsGFP_whirls.mvd2-20190704_CsGFP_d17_whirls 14_frame1.tif", -1)
im2 = cv2.imread("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/data/train_frames/train/bordered_cropped_C1-20190704_CsGFP_whirls.mvd2-20190704_CsGFP_d17_whirls 14_frame1.tif", -1)
ma = cv2.imread("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/data/train_masks/train/bordered_C1-20190704_CsGFP_whirls.mvd2-20190704_CsGFP_d17_whirls 14_frame1_mask.tif")
ma2 = cv2.imread("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/data/train_masks/train/bordered_C1-20190704_CsGFP_whirls.mvd2-20190704_CsGFP_d17_whirls 14_frame1_mask.tif")
ma = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
ma2 = cv2.cvtColor(ma2, cv2.COLOR_BGR2GRAY)
"""


