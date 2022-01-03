# Author : Ghjulia

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.model_operations import *
from utils.image_operations import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_path = '../../../scratch/gsialelli/DigitWdb'
save_path = 'saved_models/'

# Whether to apply random feature extraction
random_features = False
# Whether to apply PCA feature extraction
PCA_cond = False
# Whether to reshuffle the data, as described in the report
reshuffled = True
# Whether to print informative messages 
VERBOSE = True
# Whether to only consider the Conv2d layers
only_conv = True

###########################################################################################
# Helper functions ########################################################################
###########################################################################################


def loadModelWeights():
    """    
        Return two dataframes (train and test) with the weights of all models, per layer.
    """
    bias = ['0.02', '0.03', '0.04', '0.05']
    train_df, train_modelId = pd.DataFrame(), 0
    test_df, test_modelId = pd.DataFrame(), 0
    
    for b in bias:
        train_data = ModelDataset(bias=b, data_directory=data_path + '/train')
        test_data = ModelDataset(bias=b, data_directory=data_path + '/test')
        
        if reshuffled : 
            train_data, test_data = balance_datasets(train_data = train_data, test_data = test_data, split1 = [int(0.7*len(train_data)), int(0.3*len(train_data))], split2=[int(0.7*len(test_data)), int(0.3*len(test_data))])
            
        for modelNumber in tqdm(range(len(train_data)), desc="loading model weights with bias "+b):
            model = train_data[modelNumber]
            layerNumber = 0

            if only_conv : layers_to_consider = model.layers[:3]
            else : layers_to_consider = model.layers

            for layer in layers_to_consider:
                if len(layer.get_weights()) != 0:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    train_df = train_df.append({'modelId':train_modelId,'weights':np.ravel(weights),'biases':np.ravel(biases),'layer':layerNumber, 'bias':b}, ignore_index=True)
                    layerNumber = layerNumber + 1
            train_modelId += 1       
        
        for modelNumber in tqdm(range(len(test_data)), desc="loading model weights with bias "+b):
            model = test_data[modelNumber]
            layerNumber = 0

            if only_conv : layers_to_consider = model.layers[:3]
            else : layers_to_consider = model.layers

            for layer in layers_to_consider:
                if len(layer.get_weights()) != 0:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    test_df = test_df.append({'modelId':test_modelId,'weights':np.ravel(weights),'biases':np.ravel(biases),'layer':layerNumber, 'bias':b}, ignore_index=True)
                    layerNumber = layerNumber + 1
            test_modelId += 1     
    
    return train_df, test_df


def apply_PCA(trainModelWeights, testModelWeights, components=10):
    """
        Apply PCA feature extraction on a per-layer basis (ie. all layer_i's together).
    """

    pca_train_weights = []
    pca_test_weights = []

    for layer in range(5) :
        
        X_train = [x[0] for x in trainModelWeights[trainModelWeights['layer'] == layer][['weights']].values]
        X_test = [x[0] for x in testModelWeights[testModelWeights['layer'] == layer][['weights']].values]
        
        pca = PCA(n_components = components)
        pca_train_weights.append(pca.fit_transform(X_train))
        pca_test_weights.append(pca.transform(X_test))      

    return pca_train_weights, pca_test_weights


def get_PCA_ModelWeights(ModelWeights, pca_model_weights) :
    """
        Put the PCA weights in a format suitable for the RNN model.
    """
    
    PCA_ModelWeights = ModelWeights.copy()
    ordered_pca_weights = []

    for l0,l1,l2,l3,l4 in zip(pca_model_weights[0], pca_model_weights[1],pca_model_weights[2],pca_model_weights[3],pca_model_weights[4]) :
        ordered_pca_weights.append(l0)
        ordered_pca_weights.append(l1)
        ordered_pca_weights.append(l2)
        ordered_pca_weights.append(l3)
        ordered_pca_weights.append(l4)

    PCA_ModelWeights = PCA_ModelWeights.assign(pca_weights=ordered_pca_weights)

    return PCA_ModelWeights


def dataset_iterator(ids, ModelWeights, feature):
    """
        Actually put the dataset in the proper format for feeding to the RNN.

        -- ids : indices of the models to be considered
        -- ModelWeights : dataset containing the weights of all models
        -- feature : whether to consider `weights` only, `biases` only, or `both`
    """

    dataset = []
    labels = []

    for modelid in ids : 

        if feature == 'both' :
            we = ModelWeights[ModelWeights['modelId'] == modelid][['weights']].values[:,0]
            bi = ModelWeights[ModelWeights['modelId'] == modelid][['biases']].values[:,0]
            res = []
            for w,b in zip(we, bi) :
                r = np.concatenate((w,b))
                res.append(r)
            X = np.array(res)

        else : 
            X = ModelWeights[ModelWeights['modelId'] == modelid][[feature]].values[:,0]
        
        y = float(ModelWeights[ModelWeights['modelId'] == modelid][['bias']].values[:,0][0])

        layers = []
        for layer in X : 
            if random_features : 
                layer = np.random.choice(layer, size = 100, replace = False) 
            elif (not PCA_cond) : 
                # padding for Conv1d layer
                max_w_len = 27648
                max_b_len = 128
                if feature == 'both' : max_layer_len = max_w_len + max_b_len
                elif feature == 'weights' : max_layer_len = max_w_len
                else : max_layer_len = max_b_len
                layer = np.pad(layer, pad_width=(0, max_layer_len - len(layer)))
            layers.append(layer)
        
        dataset.append(layers)
        labels.append(y)
    
    return dataset, labels
        

def train_test(trainModelWeights, testModelWeights, feature='weights'):
    """
        Splitting the obtained RNN-ready-dataset into train/val/test sets.
    """
    
    train_ids = list(range(0, int(trainModelWeights['modelId'].max() + 1)))
    test_ids = list(range(0, int(testModelWeights['modelId'].max() + 1)))
    train_ids, val_ids = train_test_split(train_ids, test_size = 0.2)
    

    train_dataset, train_labels = dataset_iterator(train_ids, trainModelWeights, feature)
    val_dataset, val_labels = dataset_iterator(val_ids, trainModelWeights, feature)
    test_dataset, test_labels = dataset_iterator(test_ids, testModelWeights, feature)
    
    return np.array(train_dataset), np.array(train_labels), train_ids, \
        np.array(val_dataset), np.array(val_labels), val_ids, \
        np.array(test_dataset),  np.array(test_labels), test_ids


def slicer(batch_size, num_rows, shuffle) :
    """
    Batch slicer, helper function of `batcher`.
    """
    
    slices = []

    start = 0
    end = batch_size
    while end <= num_rows :
        slices.append((start,end))
        start = end
        end += batch_size
    
    if shuffle : 
        np.random.shuffle(slices)
    
    return slices


def batcher(X_dataset, y_dataset, batch_size=8, shuffle=True):
    """
    Very basic iterator.
    """

    softmax_labels = {0.02 : [1,0,0,0], 0.03 : [0,1,0,0], 0.04 : [0,0,1,0], 0.05 : [0,0,0,1]}
    num_rows = X_dataset.shape[0]
    slices = slicer(batch_size, num_rows, shuffle)

    for (start, end) in slices : 
        batch = torch.Tensor(X_dataset[start:end, :, :])
        y_CE = []
        for y_label in y_dataset[start:end] : 
            y_CE.append(softmax_labels[y_label])
        #labels = torch.Tensor(y_dataset[start:end])
        labels = torch.Tensor(y_CE)
        yield (batch, labels)


###########################################################################################
# The model ###############################################################################
###########################################################################################

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        # bidirectional – If True, becomes a bidirectional LSTM, let's not get too complicated at first
        
        # Available layers (for trying different architectures)

        # Convolutional layers
        self.conv1 = nn.Conv1d(27648, 1000, 1)
        self.conv2 = nn.Conv1d(10000, 1000, 1)
        self.conv3 = nn.Conv1d(1000, 100, 1)
        # LSTM layers
        self.lstm0 = nn.LSTM(input_size = 27648, hidden_size = 1000, num_layers = 1, batch_first = True, bidirectional = False)
        self.lstm1 = nn.LSTM(input_size = 1000, hidden_size = 100, num_layers = 1, batch_first = True, bidirectional = False)
        self.lstm2 = nn.LSTM(input_size = 1000, hidden_size = 10, num_layers = 1, batch_first = True, bidirectional = False)
        self.lstm4 = nn.LSTM(input_size = 27648, hidden_size = 100, batch_first = True)
        
        # Linear layers
        self.dense = nn.Linear(10, 4)
        self.dense2 = nn.Linear(100, 4)

        # Softmax layer
        self.sm = nn.Softmax(dim = 2)
        
        # GRU layers
        self.gru0 = nn.GRU(input_size = 27648, hidden_size = 100, batch_first = True)
        self.gru1 = nn.GRU(input_size = 27648, hidden_size = 1000, batch_first = True)
        self.gru2 = nn.GRU(input_size = 1000, hidden_size = 100, batch_first = True)

    def forward(self, x):

        # input : (N, L, H_in) 
        # where N = batch size
        #       L = sequence length
        #       H_in = input size
        # -> (batch_size, 5, num_features)

        x, (hidden, _) = self.lstm4(x)
        hidden = torch.reshape(hidden, shape = (hidden.size()[1], hidden.size()[0], hidden.size()[2]))
        x = self.dense2(hidden)
        
        return torch.reshape(x, shape = (x.size()[0],x.size()[2]))


###########################################################################################
# Execution ###############################################################################
###########################################################################################

if __name__ == "__main__" :

    save_path += 'conv_only_reshuffled_latest_10_epochs_lstm4_dense2/'
    os.mkdir(save_path)

    ##### Loading the data ####################################################################

    if VERBOSE : print('Loading weights...')
    trainModelWeights, testModelWeights = loadModelWeights()

    if PCA_cond :
        if VERBOSE : print('Extract PCA...') 
        pca_train_weights, pca_test_weights = apply_PCA(trainModelWeights, testModelWeights)
        PCA_trainModelWeights = get_PCA_ModelWeights(trainModelWeights, pca_train_weights)
        PCA_testModelWeights = get_PCA_ModelWeights(testModelWeights, pca_test_weights)

    if VERBOSE : print('Split data...')
    if PCA_cond : X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids = train_test(PCA_trainModelWeights, PCA_testModelWeights, feature='pca_weights')
    else : X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids = train_test(trainModelWeights, testModelWeights, feature='weights')

    # Sanity checks : 
    if VERBOSE : 
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
        print(np.mean(X_train), np.mean(X_val), np.mean(X_test))
        print(np.std(X_train), np.std(X_val), np.std(X_test))

    ##### Training the model ###################################################################

    model = Model()

    criterion = CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    BATCH_SIZE = 16
    NUM_EPOCHS = 10


    best_loss = 10000000.0
    losses = []
    accuracy = []
    val_losses = []
    val_accuracy = []

    for epoch in range(NUM_EPOCHS):
        if VERBOSE : print('Epoch : ', epoch+1)

        epoch_loss = []
        epoch_accuracies = []

        epoch_val_loss = []
        epoch_val_accuracies = []

        for data in batcher(X_train, y_train, batch_size = BATCH_SIZE) :
            
            # pass through model
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # backprop
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            epoch_loss.append(loss.item())

            # get training accuracy
            y_true = [lab.argmax() for lab in labels]
            y_pred = [pred.argmax() for pred in outputs]
            epoch_accuracies.append(accuracy_score(y_true, y_pred))

        losses.append(np.mean(epoch_loss))
        accuracy.append(np.mean(epoch_accuracies))

        # Validation accuracies and losses
        for data in batcher(X_val, y_val, batch_size=y_val.shape[0]) :
            inputs, labels = data
            outputs = model(inputs)

            # test loss
            loss = criterion(outputs, labels)
            val_loss = loss.item()
            epoch_val_loss.append(val_loss)

            # validation accuracy
            y_true = [lab.argmax() for lab in labels]
            y_pred = [pred.argmax() for pred in outputs]
            val_acc = accuracy_score(y_true, y_pred)
            epoch_val_accuracies.append(val_acc)
        
        val_losses.append(np.mean(epoch_val_loss))
        val_accuracy.append(np.mean(epoch_val_accuracies))

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        fn = save_path + 'model.pth.tar'
        if is_best :
            fn = save_path + 'best_model.pth.tar'
            if VERBOSE : print('Saving new best model') 
        state = {'epoch' : epoch + 1, 'state_dict' : model.state_dict(), 'best_loss' : best_loss, 'optimizer' : optimizer.state_dict()}
        torch.save(state, fn)


    plt.plot(list(range(len(losses))), losses, label = 'Training')
    plt.plot(list(range(len(losses))), val_losses, label = 'Validation')
    plt.title('Loss')
    plt.legend()
    plt.savefig(save_path + 'loss.png')
    plt.show()

    """
    plt.plot(list(range(len(accuracy))), accuracy, label = 'Training')
    plt.plot(list(range(len(accuracy))), val_accuracy, label = 'Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(save_path + 'accuracy.png')
    plt.show()
    """

    ##### Testing time #########################################################################

    best_model_info = torch.load(save_path + 'best_model.pth.tar')
    best_model = Model()
    best_model.load_state_dict(best_model_info['state_dict'])

    for data in batcher(X_test, y_test, batch_size=y_test.shape[0]) :
        inputs, labels = data
        outputs = best_model(inputs)

        # test loss
        loss = criterion(outputs, labels)

        # test accuracy
        y_true = [lab.argmax() for lab in labels]
        y_pred = [pred.argmax() for pred in outputs]
        test_accuracy = accuracy_score(y_true, y_pred)

    if VERBOSE : 
        print()
        print('save path : ', save_path)
        print('Test loss : ', loss.item())
        print('Test accuracy : ', test_accuracy)
