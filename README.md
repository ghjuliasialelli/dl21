# Deep Learning Project Repository

## Group Members
- Ghjulia Sialleli
- Luca Schweri
- David Scherer
- Philip Toma

---
## Usage Instructions

Different sections of the report were generated with different files, which all are supplied in this repository.
__First__, the requirements, which need to be installed / upgraded:
- torch
- pytorch-lightning
- sklearn
- tensorflow

### Bias Classifiers
To obtain the bias-classifiers, it is necessary to reproduce (i.e. train the classifiers). To do this, run:
```
python trainer.py --epochs [num_epochs] --stepsize [1 for 100% train data, 10 for 10%] --model_number [number] [path-to-data]
```
Furthermore, it is possible to supply a model number, where:
- 0 corresponds to Dense
- 1 corresponds to Dense+Dense
- 2 corresponds to Conv
- 3 corresponds to Conv+Dense

The models will be stored in the directory [path-to-data]/bias_classifiers.

For refinement of the classifiers, use:
```
python trainer.py --epochs [num_epochs] --stepsize [1 for 100% train data, 10 for 10%] [path-to-data]
```

### Generalization Dataset
To generate data as we did for the generalization dataset, use:
```
python3 MNIST_trainer.py --epochs 1 [path-to-colored_mnist] [path-to-store-the-generalization_dataset]```
```

### Principal Component Analysis and K-Nearest Neighbors

To run the PCA+KNN baseline use the following command:
```
python test_baseline.py
```
The program prints and plots the accuracy for different datasets when using PCA+KNN.

### Bias Mitigation

In our project we have used different bias localization and mitigation methods. To run the bias mitigation use the following command:
```
python bias_mitigation.py
```
The methods used and the hyperparameters can be changed in [bias_mitigation_options.json](options/bias_mitigation_options.json).
The current option file contains all possible methods and the following is a list of all methods and their hyperparameters:
- **Bias Localization Methods**
  - **Fast Gradient Sign Method** (fgsm)
    - *epsilon*: The magnitude of change in each step
    - *steps*: Number of steps to execute
    - *neuron_threshold*: Threshold used for neuron-wise localization of the bias
  - **Gradient Method** (gradient)
    - *epsilon*: The maximal magnitude of change in each step
    - *steps*: Number of steps to execute
    - *neuron_threshold*: Threshold used for neuron-wise localization of the bias
  - **Permutation Importance** (perm_imp)
    - *iterations*: Number of permutation iterations executed per layer
- **Bias Mitigation methods**
  - **Fast Gradient Sign Method** (fgsm)
    - *epsilon*: The magnitude of change in each step
    - *steps*: Number of steps to execute
  - **Gradient Method** (gradient)
    - *epsilon*: The maximal magnitude of change in each step
    - *steps*: Number of steps to execute
  - **Reset Weights Layer-Wise** (layer_reset)
  - **Reset Weights Neuron-Wise** (neuron_reset)
  - **Dropout** (dropout)
    - *prob*: The dropout probability used in the added dropout layer
  - **Fine-Tune** (fine_tune)
    - *lr*: The learning rate used in fine-tuning
    - *epochs*: The number of epochs to fine-tune

You can find more information about these methods in the report.



