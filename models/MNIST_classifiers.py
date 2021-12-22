import torch
from torch.nn import Module, Conv2d, MaxPool2d, Flatten, ReLU, Softmax, Linear, Dropout


class MNISTClassifier(Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        self.maxpool = MaxPool2d(kernel_size=(2, 2))
        self.conv1 = Conv2d(3, 24, kernel_size=(5, 5))
        self.conv2 = Conv2d(24, 48, kernel_size=(3, 3))
        self.conv3 = Conv2d(48, 64, kernel_size=(3, 3))
        self.flatten = Flatten()
        self.dense1 = Linear(64, 128)
        self.dropout = Dropout(0.3)
        self.dense2 = Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.softmax(self.dense2(x))
        return x

    def set_weights(self, weights):
        state_dict = self.state_dict()
        state_dict['conv1.weight'] = weights['layer_0'][0]
        state_dict['conv1.bias'] = weights['layer_1'][0, :, 0, 0, 0]
        state_dict['conv2.weight'] = weights['layer_2'][0]
        state_dict['conv2.bias'] = weights['layer_3'][0, :, 0, 0, 0]
        state_dict['conv3.weight'] = weights['layer_4'][0]
        state_dict['conv3.bias'] = weights['layer_5'][0, :, 0, 0, 0]
        state_dict['dense1.weight'] = weights['layer_6'][0, :, :, 0, 0]
        state_dict['dense1.bias'] = weights['layer_7'][0, :, 0, 0, 0]
        state_dict['dense2.weight'] = weights['layer_8'][0, :, :, 0, 0]
        state_dict['dense2.bias'] = weights['layer_9'][0, :, 0, 0, 0]
        self.load_state_dict(state_dict, strict=True)

"""
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf

#input_shape is (height, width, number of channels) for images
# include_top=False allows us to easily change the final layer to our custom dataset.
input_shape = (32, 32, 3)
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

# Prepare Model:
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))

#avoid overfitting
#model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))

# Set NUMBER_OF_CLASSES = 10 for MNIST-classification.
NUMBER_OF_CLASSES = 10
model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
conv_base.trainable = False

print(model.summary())

ws = model.get_weights()
print(ws[len(model.get_weights())-2].shape)
# Hypothesis: Biased training will lead to bias effecting the dense layer, which for the EfficientNet is:
"""