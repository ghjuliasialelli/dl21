# Script to cluster the models 

# pip install torch torchvision
# pip install --upgrade tensorflow
# pip install scikit-learn matplotlib tqdm

from model_operations import ModelDataset
from image_operations import DigitData_Torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from os.path import isfile


def test_all_models(PATH_TO_DATA = '../../../../scratch/gsialelli'):
    # on Ghjulia's computer : 
    # PATH_TO_DATA = '../data'
    # on Ghjulia's cluster :
    # PATH_TO_DATA = '../../../../scratch/gsialelli'

    # for plotting 
    jitters, accuracies = [], []

    for jitter in ['0.02', '0.03', '0.04', '0.05'] :

        model_dataset = ModelDataset(bias=jitter, data_directory='{}/DigitWdb/train/'.format(PATH_TO_DATA))
        mnist_dataset = DigitData_Torch(path='{}/colored_mnist/'.format(PATH_TO_DATA), cj_variance = jitter, mode='test')
        images = mnist_dataset.images
        labels = mnist_dataset.labels

for jitter in ['0.02', '0.03', '0.04', '0.05']:

    model_dataset = ModelDataset(bias=jitter, data_directory='../data/DigitWdb/train/')
    mnist_dataset = DigitData_Torch(path='../data/colored_mnist/', cj_variance=jitter, mode='test')
    images = mnist_dataset.images
    labels = mnist_dataset.labels

    for i in range(model_dataset.num_models):
        model = model_dataset[i]
        bulk_pred = model.predict(images)
        pred_labels = np.argmax(bulk_pred, axis=1)
        acc = accuracy_score(labels, pred_labels)

        # for plotting
        jitters.append(float(jitter))
        accuracies.append(acc)

    return jitters, accuracies


def plot_results(save = False):
    if (isfile('jitters.npy') and isfile('accuracies.npy')) : 
        jitters = np.load('jitters.npy')
        accuracies = np.load('accuracies.npy')

        plt.scatter(jitters, accuracies)
        plt.xlabel('Color jitter')
        plt.ylabel('Accuracy')
        plt.title('Models accuracies as a function of color jitter variance')
        if save : plt.savefig('jitter_vs_acc.png')
        plt.show()

    else : print('Please execute `test_all_models()`.')