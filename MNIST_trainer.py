# ------------
# Author:       Philip Toma
# Description:  This file trains an arbitrary MNIST-classifier architecture (written in tensorflow).
# Usage:        python3 MNIST_trainer.py --epochs num_epochs --num_models num_models path_to_colored_MNIST save_path
# Data:         https://github.com/feidfoe/learning-not-to-learn/tree/master/dataset/colored_mnist
# ------------
# /cluster/home/tomap/dl21/data

from models.MNIST_classifiers import *

"""from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
from sklearn import model_selection"""
# from keras import optimizers
from tensorflow.keras import optimizers
import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import matplotlib.pyplot as plt


# ------------------------------------------------
# Helper Functions
# ------------------------------------------------

def plot_losses(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def highest_model_number(path):
    ls = []
    for bias in ['0.020', '0.030', '0.040', '0.050']:
        # get highest model number in directory
        max_i = 0
        model_names = os.listdir(os.path.join(path, bias))
        for m in model_names:
            i = int(m[6:-3])
            if i > max_i:
                max_i = i
        ls.append(max_i)
        max_i = 0
    return min(ls)

# ------------------------------------------------
# Parameter Initialisation
# ------------------------------------------------

parser = argparse.ArgumentParser(description='Run MNIST-Classifier Training.')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs for model training.')
parser.add_argument('--num_models', type=int, default=1,
                    help='number of models to train.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for lower computational impact.')
parser.add_argument('--gpu', action='store_true',
                    help='set --gpu flag for gpu computation.')
parser.add_argument('path', type=str, default='./data/colored_mnist',
                    help='path to the MNIST data.')
parser.add_argument('save_path', type=str, default='./data/generalization_dataset',
                    help='path to save the trained model data.')
args = parser.parse_args()

IMAGES_PATH = args.path
SVAE_PATH = args.save_path
#IMAGES_PATH = '/home/phil/Documents/Studium/DL/Project/colored_mnist/'
#SVAE_PATH = '/home/phil/Documents/Studium/DL/Project/generalization_dataset/'
batch_size = 12
os.makedirs(SVAE_PATH, exist_ok=True)

# Actual shape of MNIST is (28, 28, 3). Need to rescale. Or use padding.
input_shape = (32, 32, 3)


# ------------------------------------------------
# Training Loop of MNIST-Classifier
# ------------------------------------------------

if args.gpu:
    device = '/GPU:0'
else:
    device = '/CPU:0'

# get the highest model number to start at, to avoid overwriting
start_i = highest_model_number(SVAE_PATH) + 1

with tf.device(device):
    for i in range(start_i, start_i + args.num_models):
        for bias in ['0.020', '0.030', '0.040', '0.050']:
            # make directory to save the model_weights.
            print(f'Make directory for bias {bias}: {os.path.join(SVAE_PATH, bias)}')
            os.makedirs(os.path.join(SVAE_PATH, bias), exist_ok=True)

            data = np.load(f'{IMAGES_PATH}mnist_10color_jitter_var_{bias}.npy',
                           encoding='latin1', allow_pickle=True).item()
            # Do we need to use stratify?
            X_train, Y_train = (data['train_image'],
                                tf.keras.utils.to_categorical(
                                    data['train_label'], num_classes=10, dtype='uint8')
                                )
            # or tf.image.resize_with_pad()
            X_test, Y_test = (tf.image.resize(data['test_image'], (32, 32)),
                              tf.convert_to_tensor(tf.keras.utils.to_categorical(
                                  data['test_label'], num_classes=10, dtype='uint8')
                              ))
            train_im, valid_im, train_lab, valid_lab = train_test_split(X_train, Y_train, test_size=0.20, random_state=40,
                                                                        stratify=Y_train, shuffle=True)
            train_im = tf.image.resize(train_im, (32, 32))
            valid_im = tf.image.resize(valid_im, (32, 32))
            # tf.image.per_image_standardization(train_im)
            # tf.image.per_image_standardization(valid_im)
            training_data = tf.data.Dataset.from_tensor_slices((train_im, train_lab))
            validation_data = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab))
            test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
            autotune = tf.data.AUTOTUNE
            train_data_batches = training_data.shuffle(buffer_size=40000).batch(128).prefetch(buffer_size=autotune)
            valid_data_batches = validation_data.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=autotune)
            # print(f'checktypes;\n{type(training_data)}')

            # ------------------------------------------------
            # Initialise the MNIST classifier
            # ------------------------------------------------

            #classifier = EfficientNet_MNIST_Classifier()
            #classifier = ResNet50_MNIST_Classifier()
            classifier = Simple_MNIST_Classifier()
            classifier.model.compile(
                loss="categorical_crossentropy",
                # optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5)
                optimizer=optimizers.RMSprop(learning_rate=2e-5),
                metrics=["acc"],
            )

            """
            Callbacking: 
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                         patience=3, min_lr=1e-5, verbose=1)
            """

            # ------------------------------------------------
            # TODO: Implement batch-wise training(?)
            # ------------------------------------------------

            history = classifier.model.fit(
                train_data_batches,
                #steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // batch_size,
                epochs=args.epochs,
                validation_data=valid_data_batches,
                #validation_steps=NUMBER_OF_VALIDATION_IMAGES // batch_size,
                verbose=1,
                use_multiprocessing=True,
                workers=4,
                # callbacks=[reduce_lr]
            )

            # plot_losses(history)

            # save model weights...
            classifier.model.save_weights(filepath=f'{os.path.join(SVAE_PATH, bias)}/model_{i}.h5')

            if args.debug:
                break

        if args.debug:
            break
