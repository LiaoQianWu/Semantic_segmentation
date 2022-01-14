import numpy as np
import os
import cv2
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

###########################################################################
# Acknowledgements:
# The code was taken and adapted from Rok Mihevc (rok/unet_weight_map.py).
# https://gist.github.com/rok/5f4314ed3c294521456c6afda36a3a50
###########################################################################


def UnetWeightMap(mask, wc=None, w0=10, sigma=5):
    
    """
    Generate weight maps as specified in the U-Net paper for boolean mask.
    
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
    
    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.
    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """
    
    mask_with_labels = label(mask)
    no_label_parts = mask_with_labels == 0
    label_ids = np.unique(mask_with_labels)[1:]
    
    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(mask_with_labels != label_id)
        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        weight_map = w0 * np.exp(-1/2 * ((d1+d2)/sigma) ** 2) * no_label_parts
        weight_map = weight_map + np.ones_like(weight_map)
        
        if wc:
            class_weights = np.zeros_like(mask)
            for k, v in wc.items():
                class_weights[mask == k] = v
                weight_map = weight_map + class_weights
                
    else:
        weight_map = np.zeros_like(mask)
    return weight_map


"""
#TEST
classLabel = np.unique(label(ma))
weightMap = UnetWeightMap(ma)

plt.imshow(ma, cmap="gray")
plt.axis("off")

fig, ax = plt.subplots()
plt.axis("off")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

im = ax.imshow(weightMap, cmap="hot")

fig.colorbar(im, cax=cax, orientation="vertical")
plt.show()
"""
