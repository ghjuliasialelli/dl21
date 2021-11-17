# Author : Ghjulia
# Feature importance with XGBoost

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.model_operations import *
from utils.image_operations import *
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from tqdm import tqdm

# pip install xgboost
# pip install pandas

directory_path = '.'

def loadModelWeights(NUM_LAYERS=10):
    """
    returns dataframe with the weights of all models by layers
    """

    biases = ['0.02', '0.03', '0.04', '0.05']
    cols = ['modelId'] + ['layer{}_weights'.format(layer) for layer in range(NUM_LAYERS)] + ['bias']
    df = pd.DataFrame(columns = cols)
    modelId = 0

    for bias in biases:
        model_data = ModelDataset(bias=bias, data_directory=directory_path+'/data/DigitWdb/train')
        
        for model_number in tqdm(range(len(model_data)), desc="loading model weights with bias "+bias):
            model_weights = model_data[model_number].get_weights()
            data = pd.DataFrame([[modelId] + [np.ravel(model_weights[layer]) for layer in range(len(model_weights))] + [float(bias)]], columns = cols)
            df = df.append(data, ignore_index=True)
            modelId += 1       
    
    return df, ['layer{}_weights'.format(layer) for layer in range(NUM_LAYERS)]
            
modelWeights, layersCols = loadModelWeights()
X = modelWeights.loc[:, layersCols].to_numpy()
y = modelWeights.loc[:, layersCols].to_numpy()

model = XGBClassifier()
model.fit(X, y)

# doesnt work because : 
# - each x is not a numerical value but an array
# - y is not a class number (int from 0,..,num_classes-1)

# -> do some PCA or feature extraction to turn the arrays into 1 value -> seems bad
# or use XGBoost for finer grain feature importance, ie. for 

# feature importance
print(model.feature_importances_)
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()