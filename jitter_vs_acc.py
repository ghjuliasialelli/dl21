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
import tensorflow as tf


def test_all_models(PATH_TO_DATA = '../../../../scratch/gsialelli', test = False, data='DigitWB'):
    # on Ghjulia's computer : 
    # PATH_TO_DATA = 'data'
    # on Ghjulia's cluster :
    # PATH_TO_DATA = '../../../../scratch/gsialelli'

    # for plotting 
    jitters, accuracies = [], []
    if data == 'DigitWB' : 
        data_directory = PATH_TO_DATA + '/DigitWdb/'
        new_model = False
    else : 
        data_directory = PATH_TO_DATA + '/generalization_dataset/'
        new_model = True

    for jitter in ['0.02', '0.03', '0.04', '0.05'] :
        print(jitter)

        if test : model_dataset = ModelDataset(bias=jitter, data_directory = data_directory + 'test/', new_model=new_model)
        else : model_dataset = ModelDataset(bias=jitter, data_directory = data_directory + 'train/', new_model=new_model)
        mnist_dataset = DigitData_Torch(path='{}/colored_mnist/'.format(PATH_TO_DATA), cj_variance = jitter, mode='test')
        images = mnist_dataset.images
        images = tf.image.resize(images, (32, 32))
        labels = mnist_dataset.labels

        for i in tqdm(range(model_dataset.num_models)):
            model = model_dataset[i]
            bulk_pred = model.predict(images)
            pred_labels = np.argmax(bulk_pred, axis=1)
            acc = accuracy_score(labels, pred_labels)

            # for plotting
            jitters.append(float(jitter))
            accuracies.append(acc)

    fp = 'results/test_' if test else 'results/'
    if data != 'DigitWB' : fp += 'gen_'

    np.save(fp + 'jitters.npy', jitters)
    np.save(fp + 'accuracies.npy', accuracies)
    
    return jitters, accuracies


def plot_results(save = False, test = False, data='DigitWB'):
    fp = 'results/test_' if test else 'results/'
    if data != 'DigitWB' : fp += 'gen_'
    
    if (isfile(fp + 'jitters.npy') and isfile(fp + 'accuracies.npy')) : 
        jitters = np.load(fp + 'jitters.npy')
        accuracies = np.load(fp + 'accuracies.npy')

        plt.scatter(jitters, accuracies)
        plt.xlabel('Color jitter')
        plt.ylabel('Accuracy')
        plt.title('Models accuracies as a function of color jitter variance')
        if save : plt.savefig(fp + 'jitter_vs_acc.png')
        plt.show()

    else : print('Please execute `test_all_models()`.')