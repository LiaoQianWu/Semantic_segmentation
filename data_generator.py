import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_augmentation import *
from unet_weight_map import UnetWeightMap


IMAGE_PATH = "path/to/data/frames/"
MASK_PATH = "path/to/data/masks/"

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, images, masks=None, batch_size=8, image_size=(256, 256), shuffle=True,
                 data_aug=False, weight_map=False, prediction=False):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.data_aug = data_aug
        self.weight_map = weight_map
        self.prediction = prediction
        self.on_epoch_end()


    def __len__(self):
        return len(self.images) // self.batch_size

    
    def __getitem__(self, index):
        image_idx = self.image_idx[index * self.batch_size: (index + 1) * self.batch_size]
        temp_images = [self.images[i] for i in image_idx]
        if self.weight_map == False:
            x, y = self.__data_generation(temp_images)
            return x, y
        else:
            x, y, w = self.__data_generation(temp_images)
            return x, y, w

    
    def on_epoch_end(self):
        self.image_idx = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.seed(123)
            np.random.shuffle(self.image_idx)

            
    def __data_generation(self, temp_images):
        image_batch = []
        mask_batch = []
        if self.weight_map:
            weight_maps = []
        for i in temp_images:
            im = cv2.imread(IMAGE_PATH + i, -1) #For 16-bit images
            resized_im = cv2.resize(im, self.image_size, interpolation=cv2.INTER_NEAREST)
            resized_im = np.reshape(resized_im, (*resized_im.shape, 1))
            
            if self.prediction == False:
                ma = cv2.imread(MASK_PATH + self.masks[i])
                ma = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
                resized_ma = cv2.resize(ma, self.image_size, interpolation=cv2.INTER_NEAREST)
                resized_ma = np.reshape(resized_ma, (*resized_ma.shape, 1))
            
            if self.data_aug == True:
                toggle = np.random.randint(2)
                resized_im, resized_ma = horizontal_flip(resized_im, resized_ma, toggle)
                toggle2 = np.random.randint(2)
                resized_im, resized_ma = vertical_flip(resized_im, resized_ma, toggle2)
                resized_im, resized_ma = rotation(resized_im, resized_ma, 45)
                
            if self.weight_map:
                ma_for_map = np.reshape(resized_ma, (self.image_size))
                w_map = UnetWeightMap(ma_for_map)
                w_map = np.reshape(w_map, (*w_map.shape, 1))
                weight_maps.append(w_map)
                
            input_im = resized_im / (2**16-1)
            image_batch.append(input_im)
            
            if self.prediction == False:
                input_ma = resized_ma / 255
                mask_batch.append(input_ma)
                
        if self.weight_map == False:
            return np.array(image_batch), np.array(mask_batch)
        else:
            return np.array(image_batch), np.array(mask_batch), np.array(weight_maps)
    
