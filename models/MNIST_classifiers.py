# ------------
# Author:       Philip Toma
# Description:  This file implements a new classifier architecture (in tensorflow) for the
# Usage:        
# Reading:      https://towardsdatascience.com/an-in-depth-efficientnet-tutorial-using-tensorflow-how-to-use-efficientnet-on-a-custom-dataset-1cab0997f65c
# ------------

from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf

class MNIST_Classifier():

    def __init__(self, input_shape, model_architecture):
        model_base = model_architecture()


class EfficientNet_MNIST_Classifier():

    def __init__(self, input_shape=(32, 32, 3), NUMBER_OF_CLASSES=10):
        conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        # rescale_layer = tf.keras.layers.Rescaling(1./255, offset=0.0)
        self.model = models.Sequential()
        # self.model.add(rescale_layer)
        self.model.add(conv_base)
        self.model.add(layers.GlobalMaxPooling2D(name="gap"))

        # avoid overfitting
        # self.model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))
        self.model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
        conv_base.trainable = False

        # print(self.model.summary())

# ws = model.get_weights()
# print(ws[len(model.get_weights())-2].shape)
# Hypothesis: Biased training will lead to bias effecting the dense layer, which for the EfficientNet is:
# ws[len(model.get_weights())-2 : ]