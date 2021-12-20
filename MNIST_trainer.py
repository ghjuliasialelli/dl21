# ------------
# Author:       Philip Toma
# Description:  This file trains an arbitrary MNIST-classifier architecture.
# ------------

from models.MNIST_classifiers import *

from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf


TRAIN_IMAGES_PATH = './data/train'
VAL_IMAGES_PATH = './data/test'

batch_size = 12

test_datagen = ImageDataGenerator(rescale=1.0 / 255)train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    TRAIN_IMAGES_PATH,
    # All images will be resized to target height and width.
    target_size=(32, 32),
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
)
validation_generator = test_datagen.flow_from_directory(
    VAL_IMAGES_PATH,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode="categorical",
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=["acc"],
)