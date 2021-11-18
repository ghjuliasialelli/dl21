# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:39:47 2021

@author: Ignacio
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.data import Dataset
from utils.image_operations import DigitData_Torch
import torch

test_dir = '/home/phil/Documents/Studium/DL/Project/train'
train_dir = '/home/phil/Documents/Studium/DL/Project/test'

# !! REMOVE !!
data_dir = '/home/phil/Documents/Studium/DL/Project/colored_mnist/'

def vec_to_digit(x):
    i = 0
    while i < 10:
        if x[i] == 1.0:
            return i + 1
        i = i + 1

###############################################################################
#                       DIGIT CLASSIFIER MODEL

# Create the TF - Keras models architecture
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

print('Finished loading models')

# Print the models weights to check if they exist.
# print(models.get_weights())

db = DigitData_Torch(path=data_dir, cj_variance=('strong', "0.020"), mode='test')
tfdataset = Dataset.from_tensor_slices((db.images, db.labels)).batch(32)

print(db.images[0].shape)
#print(models(torch.from_numpy(db.images[0]).unsqueeze(dim=0).numpy()))
print(db.images.shape)

out = model(db.images)
ls = []
for k in range(0, out.shape[0]):
    ls.append(vec_to_digit(out[k]))

out = np.asarray(ls)
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