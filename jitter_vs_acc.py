# Author : Ghjulia
# - Script to show the relationship between models' jitter variances and accuracies

# pip install torch torchvision
# pip install --upgrade tensorflow
# pip install scikit-learn matplotlib tqdm

from utils.model_operations import ModelDataset
from utils.image_operations import DigitData_Torch
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

        for i in range(model_dataset.num_models):
            model = model_dataset[i]
            bulk_pred = model.predict(images)
            pred_labels = np.argmax(bulk_pred, axis=1)
            acc = accuracy_score(labels, pred_labels)

            # for plotting
            jitters.append(float(jitter))
            accuracies.append(acc)


    np.save('results/jitters.npy', jitters)
    np.save('results/accuracies.npy', accuracies)
    
    return jitters, accuracies


def plot_results(save = False):
    if (isfile('results/jitters.npy') and isfile('results/accuracies.npy')) : 
        jitters = np.load('results/jitters.npy')
        accuracies = np.load('results/accuracies.npy')

        plt.scatter(jitters, accuracies)
        plt.xlabel('Color jitter')
        plt.ylabel('Accuracy')
        plt.title('Models accuracies as a function of color jitter variance')
        if save : plt.savefig('results/jitter_vs_acc.png')
        plt.show()

    else : print('Please execute `test_all_models()`.')