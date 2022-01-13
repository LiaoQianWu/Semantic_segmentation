import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from data_generator import DataGenerator
import model
from CustomModel import CustomModel

# GPU TOGGLE
"""
# Turn off GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Check GPU status
if tf.test.gpu_device_name():
    print("GPU found")
else:
    print("No GPU found")
"""


#HYPERPARAMETER FOR DATA GENERATOR
DATA_PATH = "path/to/data" #Data allocation is described in README

params = dict(batch_size = 8,
              image_size = (608, 608),
              shuffle = True,
              data_aug=False,
              weight_map=False)

#DATA GENERATOR
train_images = os.listdir(DATA_PATH + "/train_frames/train")
train_images.sort(key=lambda filename: [int(i) if i.isdigit() else i
                                        for i in re.findall(r"[^0-9]+|[0-9]+", filename)])

val_images = os.listdir(DATA_PATH + "/val_frames/val")
val_images.sort(key=lambda filename: [int(i) if i.isdigit() else i
                                      for i in re.findall(r"[^0-9]+|[0-9]+", filename)])

train_masks = os.listdir(DATA_PATH + "/train_masks/train")
train_masks.sort(key=lambda filename: [int(i) if i.isdigit() else i
                                       for i in re.findall(r"[^0-9]+|[0-9]+", filename)])

val_masks = os.listdir(DATA_PATH + "/val_masks/val")
val_masks.sort(key=lambda filename: [int(i) if i.isdigit() else i
                                     for i in re.findall(r"[^0-9]+|[0-9]+", filename)])

test_images = os.listdir(DATA_PATH + "/test_frames")
test_images.sort(key=lambda filename: [int(i) if i.isdigit() else i
                                       for i in re.findall(r"[^0-9]+|[0-9]+", filename)])

test_masks = os.listdir(DATA_PATH + "/test_masks")
test_masks.sort(key=lambda filename: [int(i) if i.isdigit() else i
                                      for i in re.findall(r"[^0-9]+|[0-9]+", filename)])

images = dict(train = train_images, val = val_images, test = test_images)
masks = dict()
train_pair = list(zip(train_images, train_masks))
val_pair = list(zip(val_images, val_masks))
test_pair = list(zip(test_images, test_masks))
mask_pair = train_pair + val_pair + test_pair
for i in range(len(mask_pair)):
    masks[mask_pair[i][0]] = mask_pair[i][1]

train_generator = DataGenerator(images["train"], masks, **params)
val_generator = DataGenerator(images["val"], masks, **params)


#HYPERPARAMETER FOR MODEL TRAINING
weights_path = "path/to/saving/weights.{epoch:02d}-{val_loss:.2f}.h5"
num_of_epochs = 60
num_of_training_images = len(train_images)
num_of_val_images = len(val_images)

#MODEL TRAINING
########################################################################################################################
# Important:
# If 'weight_map' is set to 'True' in DataGenerator, 'custom_model' in U-net model should be set to 'True'.
# Note that optimizer status should have been saved in h5 format file, so do not have to specify optimizer again.
########################################################################################################################
model = model.unet(input_size=(608, 608, 1), custom_model=False)
# If trained model is going to be ued
#model = load_model("trained_model.h5", custom_objects={"CustomModel": CustomModel})

model.compile(loss="binary_crossentropy",
              optimizer = Adam(lr=4e-4),
              metrics = ["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)])

callbacks_list = [ModelCheckpoint(weights_path, monitor="val_loss",
                                  save_best_only=True, mode="min"),
                  EarlyStopping(monitor="val_loss", min_delta=0, patience=4,
                                mode="min", restore_best_weights=True)]

model.fit(train_generator, epochs=num_of_epochs,
          steps_per_epoch=(num_of_training_images//params["batch_size"]),
          validation_data=val_generator,
          validation_steps=(num_of_val_images//params["batch_size"]),
          verbose=1, callbacks=callbacks_list)

model.save("trained_model.h5")


"""
#EVALUTION
test_generator = DataGenerator(images["test"], masks, batch_size=1, image_size=(608, 608), shuffle=False)
                                      
model = load_model("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/Unet/(with aug)Malaria_segmentation.h5")
score = model.evaluate(test_generator, verbose=1, return_dict=True)
"""

"""
#PREDICTION
test_generator = DataGenerator(images["test"], batch_size=1, image_size=(608, 608), shuffle=False, prediction=True)

model = load_model("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/Unet/(weight map)Malaria_segmentation.h5",
                   custom_objects={"CustomModel": CustomModel})
result = model.predict(test_generator, verbose=1)

for i, name in enumerate(images["test"]):
    im = result[i]
    io.imsave("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/Unet/results/result_" + name, im, plugin="tifffile")
"""
