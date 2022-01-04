# Deep Learning Project Repository

## Group Members
- Ghjulia Sialleli
- Luca Schweri
- David Scherer
- Philip Toma

---
## Usage Instructions

Different sections of the report were generated with different files, which all are supplied in this repository.
__First__,To install dependencies, use the file ***.env for installation of a new conda environment with the required
packages.

To reproduce the classifiers, run the supplied run_models.sh file. __(??)__


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

The appriopriate flags are described and can be set at the top of the file. Then, one needs only to run : 
```
python rnn.py
```

Checkpoints of trained models can be found in the `data/trained/lstm/` folder. 

---
## Links and References

- 
