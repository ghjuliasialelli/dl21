import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.model_operations import *
from utils.image_operations import *
from sklearn.decomposition import PCA
directory_path = os.getcwd()
data_path = '../../../scratch/gsialelli/DigitWdb'

save_path = 'saved_models/'

PCA_cond = False

reshuffled = False

def loadModelWeights(percentage=1):
    """
    setname: either "train" or "test"
    
    returns dataframe with the weights of all models by layers
    """
    bias = ['0.02', '0.03', '0.04', '0.05']
    train_df, train_modelId = pd.DataFrame(), 0
    test_df, test_modelId = pd.DataFrame(), 0
    
    for b in bias:
        train_data = ModelDataset(bias=b, data_directory=data_path + '/train')
        test_data = ModelDataset(bias=b, data_directory=data_path + '/test')
        
        if reshuffled : 
            train_data, test_data = balance_datasets(train_data = train_data, test_data = test_data, split1 = [int(0.7*len(train_data)), int(0.3*len(train_data))], split2=[int(0.7*len(test_data)), int(0.3*len(test_data))])
            
        for modelNumber in tqdm(range(len(train_data)//percentage), desc="loading model weights with bias "+b):
            model = train_data[modelNumber]
            layerNumber = 0
            for layer in model.layers:
                if len(layer.get_weights()) != 0:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    train_df = train_df.append({'modelId':train_modelId,'weights':np.ravel(weights),'biases':np.ravel(biases),'layer':layerNumber, 'bias':b}, ignore_index=True)
                    layerNumber = layerNumber + 1
            train_modelId += 1       
        
        for modelNumber in tqdm(range(len(test_data)//percentage), desc="loading model weights with bias "+b):
            model = test_data[modelNumber]
            layerNumber = 0
            for layer in model.layers:
                if len(layer.get_weights()) != 0:
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    test_df = test_df.append({'modelId':test_modelId,'weights':np.ravel(weights),'biases':np.ravel(biases),'layer':layerNumber, 'bias':b}, ignore_index=True)
                    layerNumber = layerNumber + 1
            test_modelId += 1     
    
    return train_df, test_df

print('Loading weights...')
trainModelWeights, testModelWeights = loadModelWeights(percentage = 100)

# Apply PCA 

def apply_PCA(trainModelWeights, testModelWeights, components=1000):

    pca_train_weights = []
    pca_test_weights = []

    for layer in range(5) :
        
        X_train = [x[0] for x in trainModelWeights[trainModelWeights['layer'] == 0][['weights']].values]
        X_test = [x[0] for x in testModelWeights[testModelWeights['layer'] == 0][['weights']].values]
        
        pca_train, pca_test = PCA(n_components = components), PCA(n_components = components)
        pca_train_weights.append(pca_train.fit_transform(X_train))
        pca_test_weights.append(pca_test.fit_transform(X_test))      

    return pca_train_weights, pca_test_weights

def get_PCA_ModelWeights(ModelWeights, pca_model_weights) :
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


if PCA_cond :
    print('Extract PCA...') 
    pca_train_weights, pca_test_weights = apply_PCA(trainModelWeights, testModelWeights)
    PCA_trainModelWeights = get_PCA_ModelWeights(trainModelWeights, pca_train_weights)
    PCA_testModelWeights = get_PCA_ModelWeights(testModelWeights, pca_test_weights)






def dataset_iterator(ids, ModelWeights, feature, label):

    dataset = []
    labels = []

    for modelid in ids : 

        X = ModelWeights[ModelWeights['modelId'] == modelid][[feature]].values[:,0]
        if label == 'one' :
            y = float(ModelWeights[ModelWeights['modelId'] == modelid][['bias']].values[:,0][0])
        else : 
            y = [float(b) for b in ModelWeights[ModelWeights['modelId'] == modelid][['bias']].values[:,0]]

        layers = []
        for layer in X : 
            # nothing because testing PCA 
            #layer = np.random.choice(layer, size = 100, replace = False) # random sampling
            if not PCA_cond : layer = np.pad(layer, pad_width=(0, 27648 - len(layer)))    # padding for Conv1d layer
            layers.append(layer)
        
        dataset.append(layers)
        labels.append(y)
    
    return dataset, labels
        
from sklearn.model_selection import train_test_split

def train_test(trainModelWeights, testModelWeights, feature='weights', label='one'):
    
    train_ids = list(range(0, int(trainModelWeights['modelId'].max() + 1)))
    test_ids = list(range(0, int(testModelWeights['modelId'].max() + 1)))
    train_ids, val_ids = train_test_split(train_ids, test_size = 0.2)
    

    train_dataset, train_labels = dataset_iterator(train_ids, trainModelWeights, feature, label)
    val_dataset, val_labels = dataset_iterator(val_ids, trainModelWeights, feature, label)
    test_dataset, test_labels = dataset_iterator(test_ids, testModelWeights, feature, label)
    
    return np.array(train_dataset), np.array(train_labels), train_ids, \
        np.array(val_dataset), np.array(val_labels), val_ids, \
        np.array(test_dataset),  np.array(test_labels), test_ids

print('Split data...')
if PCA_cond : X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids = train_test(PCA_trainModelWeights, PCA_testModelWeights, feature='pca_weights')

else : X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids = train_test(trainModelWeights, testModelWeights, feature='weights')


print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
print(np.mean(X_train), np.mean(X_val), np.mean(X_test))
print(np.std(X_train), np.std(X_val), np.std(X_test))


def slicer(batch_size, num_rows, shuffle) :
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


softmax_labels = {0.02 : [1,0,0,0], 0.03 : [0,1,0,0], 0.04 : [0,0,1,0], 0.05 : [0,0,0,1]}

# very basic iterator

def batcher(X_dataset, y_dataset, batch_size=8, shuffle=True):

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




# other resources to continue : 
# https://stackoverflow.com/questions/58251677/how-do-i-train-an-lstm-in-pytorch

class Model(nn.Module):
    # source : https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM 
    
    def __init__(self):
        super(Model, self).__init__()
        
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        # bidirectional – If True, becomes a bidirectional LSTM, let's not get too complicated at first
        #self.conv1 = nn.Conv1d(27648, 1000, 1)
        #self.conv2 = nn.Conv1d(10000, 1000, 1)
        #self.conv3 = nn.Conv1d(1000, 100, 1)
        self.lstm0 = nn.LSTM(input_size = 27648, hidden_size = 1000, num_layers = 1, batch_first = True, bidirectional = False)
        self.lstm1 = nn.LSTM(input_size = 1000, hidden_size = 100, num_layers = 1, batch_first = True, bidirectional = False)
        self.lstm2 = nn.LSTM(input_size = 1000, hidden_size = 10, num_layers = 1, batch_first = True, bidirectional = False)
        self.dense = nn.Linear(10, 4)
        self.dense2 = nn.Linear(100, 4)
        self.sm = nn.Softmax(dim = 2)
        self.lstm4 = nn.LSTM(input_size = 27648, hidden_size = 100, batch_first = True)
        self.gru0 = nn.GRU(input_size = 27648, hidden_size = 100, batch_first = True)
        self.gru1 = nn.GRU(input_size = 27648, hidden_size = 1000, batch_first = True)
        self.gru2 = nn.GRU(input_size = 1000, hidden_size = 100, batch_first = True)

    def forward(self, x):
        # input : (N, L, H_in) 
        # where N = batch size
        #       L = sequence length
        #       H_in = input size
        # -> (batch_size, 5, num_features)
        
        # some conv1 layer to reduce the number of features

        #x = torch.reshape(x, (x.size()[0], x.size()[2], x.size()[1]))
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = torch.reshape(x, (x.size()[0], x.size()[2], x.size()[1]))
        #x, _ = self.lstm0(x)
        #x, _ = self.lstm0(x)
        #print(x.shape)
        #x, (hidden, cell) = self.lstm4(x)
        #x, _ = self.lstm0(x)
        x, (hidden,cell) = self.lstm4(x)
        # hidden : final hidden state for each element in the batch, shape (1, 8, 10) -> reshape into (8,1,10)
        hidden = torch.reshape(hidden, shape = (hidden.size()[1], hidden.size()[0], hidden.size()[2]))
        #print(x.size())
        x = self.dense2(hidden)
        #print(x.size())
        #x = self.sm(x)
        #print(x.size())
        
        return torch.reshape(x, shape = (x.size()[0],x.size()[2]))

save_path += '10_epochs_lstm4_dense2/' # if not '' dont forget the / at the end
os.mkdir(save_path)


# input size : (batch_size, 5, num_features) - for the 5 layers, and the features extracted
# output size : (batch_size, 1, 1) - for the prediction per model


model = Model()#.cuda()
# y = model(list(batcher(X_train, y_train))[0][0])

from torch import optim
from torch.nn import MSELoss, CrossEntropyLoss
from sklearn.metrics import accuracy_score

criterion = CrossEntropyLoss(reduction='mean')#.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.01)


BATCH_SIZE = 16
NUM_EPOCHS = 10

best_loss = 10000000.0

losses = []
accuracy = []

val_losses = []
val_accuracy = []

for epoch in range(NUM_EPOCHS):
    print('Epoch : ', epoch+1)

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
        #cat_preds = make_categorical(outputs)
        #cat_labels = make_categorical(labels)
        #epoch_accuracies.append(accuracy_score(cat_labels, cat_preds))
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

        # test accuracy
        #cat_preds = make_categorical(outputs)
        #cat_labels = make_categorical(labels)
        #epoch_val_accuracies.append(accuracy_score(cat_labels, cat_preds))
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
        print('Saving new best model') 
    state = {'epoch' : epoch + 1, 'state_dict' : model.state_dict(), 'best_loss' : best_loss, 'optimizer' : optimizer.state_dict()}
    torch.save(state, fn)


import matplotlib.pyplot as plt

plt.plot(list(range(len(losses))), losses, label = 'Training')
plt.plot(list(range(len(losses))), val_losses, label = 'Validation')
plt.title('Loss')
plt.legend()
plt.savefig(save_path + 'loss.png')
plt.show()

plt.plot(list(range(len(accuracy))), accuracy, label = 'Training')
plt.plot(list(range(len(accuracy))), val_accuracy, label = 'Validation')
plt.title('Accuracy')
plt.legend()
plt.savefig(save_path + 'accuracy.png')
plt.show()


# test accuracy : 

for data in batcher(X_test, y_test, batch_size=y_test.shape[0]) :
    inputs, labels = data
    outputs = model(inputs)

    # test loss
    loss = criterion(outputs, labels)

    # test accuracy
    y_true = [lab.argmax() for lab in labels]
    y_pred = [pred.argmax() for pred in outputs]
    test_accuracy = accuracy_score(y_true, y_pred)
    
    #cat_preds = make_categorical(outputs)
    #cat_labels = make_categorical(labels)
    #accuracy = accuracy_score(cat_labels, cat_preds)

print()
print('save path : ', save_path)
print('Test loss : ', loss.item())
print('Test accuracy : ', test_accuracy)
