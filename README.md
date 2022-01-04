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
- 1 corresponds to Conv
- 2 corresponds to the (adapted) Conv model used for refinement

Further, calling trainer.py has the options:
- Setting the flag **--use_dense** will make the classifier recognise and use the dense layers of the MNIST classifier.
- Setting the flag **--reshuffle** will force the reshuffling of the train- and test-dataset.

The models will be stored in the directory [path-to-data]/bias_classifiers. An example call would therefore be:
```
python3 trainer.py --model_number 0 --epochs 1 --stepsize 1 --reshuffle --use_dense ./data/author_dataset
```

For **refinement** of the classifiers in the list above, use:
```
python3 refiner.py --epochs [num-refinement-epochs] --model_number [number] [path-to-original-model_weights] [path-to-generalization_dataset] [path-to-bias_classifiers]
```

Further, calling refine.py has the options:
- Setting the flag **--test_on_old** will use the original (authors) test-set for training and refinement.
  - Omitting the flag will refine and test using the generalisation set.
- Setting the flag **--use_dense** will use the classifier that recognises and uses information from the dense layers of the MNIST classifier.
- Model numbers correspond to:
  - 0: Conv Model
  - 1: Dense model

An example call:
```
python3 refiner.py --epochs 1 --model_number 0 --use_dense ./data/author_dataset ./data/generalization_dataset ./data/author_dataset
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

### Bias-level Classifers
Various classifiers were investigated for this task.

#### RNN
All operations concerning the RNN model take place in the `rnn.py` file. 

The appriopriate flags are described and can be set at the top of the file. They are currently set for the training of the RNN model on the re-shuffled DigiitWdb dataset. 
Then, one needs only to run : 
```
python rnn.py
```

Checkpoints of trained models can be made available upon request. They would normally be found in a shared Polybox folder, but poor Wifi connection is preventing their uploading.

---
## Links and References

- 
