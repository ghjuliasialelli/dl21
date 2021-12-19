import umap
import torch
import numpy as np


from tqdm import tqdm
from sklearn.decomposition import PCA
from utils.model_operations import *
from utils.image_operations import *
import matplotlib.pyplot as plt


def model_to_vec(model):
    return np.concatenate([np.ravel(model[j]) for j in range(len(model))])


def build_umap(percentage):
    res_ls = []
    bias = ['0.02', '0.03', '0.04', '0.05']
    c_vals = []

    for b in bias:
        model_data = ModelDataset(bias=b, data_directory='/home/phil/Documents/Studium/DL/Project/train/')
        #print(np.stack([model_to_vec(model_data[i].get_weights()) for i in range(len(model_data))]).shape)
        ls = []
        #print(model_data.print_digit_classifier_info(0, True))
        # Build 2D array of models weights..
        # For debugging purposes: use 1% of the data with //100 or //10 (10% of data)
        for i in tqdm(range(len(model_data)//percentage), desc="Building 2D-Array"):
            #print(i/len(model_data))
            #ls.append(model_to_vec(model_data[i].get_weights()))

            # Only use first dense layer:
            ls.append(model_to_vec(model_data[i].get_weights()[7]))

            # Only use the convolutional layers:
            #ls.append(model_to_vec(model_data[i].get_weights()[:4]))
            # list of models weights (multiple layers)
            #models = model_data[i].get_weights()
            #tmp = np.concatenate([np.ravel(models[j]) for j in range(len(models))])
            c_vals.append(float(b))
        arr2d = np.stack(ls)
        print(f'Shape after 2D-conversion: {arr2d.shape}')
        res_ls.append(arr2d)

    #pca = PCA(n_components=2, svd_solver='full')
    arr2d = np.concatenate(res_ls)
    print(arr2d.shape)
    print(arr2d.max(axis=1).shape)
    #arr2d = arr2d.transpose() / arr2d.max(axis=1)
    print(f'Resulting shape: {arr2d.shape}')

    fit = umap.UMAP()
    #u = fit.fit_transform(arr2d.transpose())
    u = fit.fit_transform(arr2d)
    print(u.shape)
    plt.scatter(u[:, 0], u[:, 1], c=c_vals)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Visualizing the Relation of Model Weights and Model Bias")
    plt.show()


# percentage = 0.1 -> 10%, 0.01 -> 1% of the models used; 1 -> 100% of the models used.
# percentage = 0.1
percentage = 1.0
build_umap(percentage=int(1/percentage))
