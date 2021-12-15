# Author : Ghjulia
# - Pipeline 

import torch
import numpy as np
from tqdm import tqdm
from utils.model_operations import *
from utils.image_operations import *
import matplotlib.pyplot as plt
import json
from os.path import isfile


########################################
# Import (encoding, classifier) pair

# Available : 
# - (PCA, KNN)
# TO DO : author's architecture, other baselines

classifier = ...

########################################
# Interpretability study

# options : 

# 1. Layer-wise ablation study w/ analysis of impact on classifier's performance

# TO DO : function that removes a given layer from the model weights
# TO DO : define metrics 


# 2. Using tools for feature importance, independent of classifier 


########################################
# Mitigating the bias

# replacing the most biased layer(s) : eg. random weights 
# dropout


########################################
# Generalization study

# train our own digit classifier (with a different architecture) on ColoredMNIST 
# and check how it reacts to the pipeline