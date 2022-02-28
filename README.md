## Semantic segmentation

Perform image segmentation on fluorescence microscopy images using [Keras](https://keras.io/api/) framework and [U-net](https://arxiv.org/pdf/1505.04597.pdf) model.

> To be improved:
* **Data allocation** (i.e., training, validation, and test data): Data splitter and data generator can be incorporated into model training.
* Functions for e.g., pre/post-processing or prediction evaluation, can be compiled into *utils.py*.
* Code optimization: Create a **module** for neatly calling model training, evaluation, or prediction.
* Explore **region-based metrics** for model performance evaluation. (Extremely important when dealing with very near objects)
