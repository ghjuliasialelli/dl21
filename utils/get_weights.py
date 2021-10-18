# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:39:47 2021

@author: Ignacio
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



test_dir = 'D:/DataSets/DigitWdb/test/'
train_dir = 'D:/DataSets/DigitWdb/train/'


###############################################################################
#                       DIGIT CLASSIFIER MODEL

# Create the TF - Keras model architecture 
model = Sequential()
model.add(Conv2D(24,kernel_size=5,activation='relu', input_shape=(28,28,3)))
model.add(MaxPooling2D())

model.add(Conv2D(48,kernel_size=3,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(MaxPooling2D())
    
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load the weights into the architecture
model.load_weights(train_dir + '/0.02/0.h5')


###############################################################################
#                       EXTRACT MODEL'S WEIGHTS

# In a CNN, each conv layer has two kinds of parameters : weights and biases
W_weigths = []
b_weights = []
for layer in model.layers:
    # We extract the parameters of the convolutional layers
    if layer.__class__.__name__ == 'Conv2D':# or layer.__class__.__name__ == 'Dense':
        W_weigths.append(layer.get_weights()[0])
        b_weights.append(layer.get_weights()[1])