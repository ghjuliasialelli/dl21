# Script to cluster the models 

# TO DO : add something that allows to load from file in utils folder
from model_operations import ModelDataset
from image_operations import DigitData_Torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json


# for plotting 
jitters, accuracies = [], []

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


np.save('jitters.npy', np.array(jitters))
np.save('accuracies.npy', np.array(accuracies))



"""
plt.scatter(jitters, accuracies)
plt.xlabel('Color jitter (inverse bias)')
plt.ylabel('Accuracy')
plt.show()
"""