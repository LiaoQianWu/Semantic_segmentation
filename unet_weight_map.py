import numpy as np
import os
import cv2
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def UnetWeightMap(mask, wc=None, w0=10, sigma=5):
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
os.chdir("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/data/train_masks/train")

ma = cv2.imread("C1-20190704_CsGFP_whirls.mvd2-20190704_CsGFP_d17_whirls 14_frame1_mask.tif")
ma = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
ma = cv2.resize(ma, (608, 608), interpolation=cv2.INTER_NEAREST)

classLabe = np.unique(label(ma))
weightMap = UnetWeightMap(ma)

#plt.imshow(ma, cmap="gray")
#plt.axis("off")

fig, ax = plt.subplots()
plt.axis("off")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

im = ax.imshow(weightMap, cmap="hot")

fig.colorbar(im, cax=cax, orientation="vertical")
plt.show()
"""