# Torch Imports:
import torch
from torch import nn

# TF Imports:
#from tensorflow.keras.applications import * #Efficient Net included here
from keras.applications import * #Efficient Net included here
# from tensorflow.keras import models
from keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, 10)

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


class EfficientNet_MNIST_Classifier():
    """New MNIST-classifier to test impact of bias on dense layer at end."""

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
        conv_base.trainable = True # Trainable of efficientNet is false to allow faster training, but leads to lower
        # accuracy!


class ResNet50_MNIST_Classifier():

    def __init__(self, input_shape=(32, 32, 3), NUMBER_OF_CLASSES=10):
        conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        # rescale_layer = tf.keras.layers.Rescaling(1./255, offset=0.0)
        self.model = models.Sequential()
        # self.model.add(rescale_layer)
        self.model.add(conv_base)
        self.model.add(layers.GlobalMaxPooling2D(name="gap"))

        # avoid overfitting
        # self.model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))
        self.model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
        conv_base.trainable = False  # Trainable of efficientNet is false to allow faster training, but leads to lower
        # accuracy!


class Simple_MNIST_Classifier():

    def __init__(self):
        self.model = Sequential()
        # self.model.add(Conv2D(24, kernel_size=5, activation='relu', input_shape=(28, 28, 3)))
        self.model.add(Conv2D(24, kernel_size=5, activation='relu', input_shape=(32, 32, 3)))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(48, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())
        self.model.add(Dense(3163, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(10, activation='softmax'))
        # self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])