import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd


# DATASETS

def load_network_weights_dataset(options):
    from utils.model_operations import LucasModelDataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return LucasModelDataset(device, ['data/digitWdb_train.pkl.gz', 'data/digitWdb_test.pkl.gz'],
                             use_weights=options['use_weights'], use_biases=options['use_weights'], only_conv=False)


def load_unbiased_color_mnist_dataset():
    from utils.image_operations import LucasDigitData
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return LucasDigitData(device, [
        # 'data/colored_mnist/mnist_10color_jitter_var_0.020.npy',
        # 'data/colored_mnist/mnist_10color_jitter_var_0.025.npy',
        # 'data/colored_mnist/mnist_10color_jitter_var_0.030.npy',
        # 'data/colored_mnist/mnist_10color_jitter_var_0.035.npy',
        # 'data/colored_mnist/mnist_10color_jitter_var_0.040.npy',
        # 'data/colored_mnist/mnist_10color_jitter_var_0.045.npy',
        'data/colored_mnist/mnist_10color_jitter_var_0.050.npy'
    ], train=False, test=True)


# NETWORKS

def load_network(options, layer_shapes):
    # create network
    if options['network'] == 'conv2d_ifbid':
        from models.ifbid import Conv2D_IFBID_Model2
        net = Conv2D_IFBID_Model2(layer_shapes, 4)
    elif options['network'] == 'rnn':
        raise NotImplementedError()
        # TODO load recurrent neural network
    else:
        raise ValueError("Network [{}] not recognized.".format(options['network']))

    # load network to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # load pretrained network weights
    state_dict = torch.load(options['pretrained_network_weights_path'], map_location=device)
    net.load_state_dict(state_dict, strict=True)

    return net


def get_digit_classifier(weights):
    from models.MNIST_classifiers import MNISTClassifier
    classifier = MNISTClassifier()
    classifier.set_weights(weights)

    # load network to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    return classifier


# BIAS METRICS

def accuracy_on_unbiased_mnist(net):
    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset()
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    net.eval()

    with torch.no_grad():
        trues = []
        preds = []
        for X, y in mnist_dataloader:
            preds.append(np.argmax(net(X).detach().cpu().numpy()[0]))
            trues.append(y.detach().cpu().numpy()[0, 0])
        preds = np.array(preds)
        trues = np.array(trues)

        accuracy = np.count_nonzero(preds == trues) / len(preds)

    return accuracy


# MITIGATION METHODS

def fast_gradient_sign_method(net, data, epsilon=2e-2, steps=1):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fcn = torch.nn.CrossEntropyLoss()

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        new_weights[k] = v.clone().requires_grad_()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    # prediction before
    with torch.no_grad():
        prediction_before = net(data['model_weights']).detach().cpu().numpy()[0]

    # FGSM steps
    for i in range(steps):
        pred = net(new_data['model_weights'])
        loss = loss_fcn(pred, torch.tensor([3], device=device).long())
        loss.backward()

        for layer in new_data['model_weights']:
            sign_grad = torch.sign(new_data['model_weights'][layer].grad.to(device))
            with torch.no_grad():
                new_data['model_weights'][layer] -= epsilon * sign_grad
            new_data['model_weights'][layer].grad.data.zero_()

    # no grad on weights
    for layer in new_data['model_weights']:
        new_data['model_weights'][layer].requires_grad_(False)

    # prediction after
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()[0]

    return new_data, prediction_before, prediction_after

# MAIN

def main():
    # load options
    with open('options/bias_mitigation_options.json', 'r') as f:
        options = json.load(f)

    # load network weights dataset
    weights_dataset = load_network_weights_dataset(options)
    weights_dataloader = DataLoader(weights_dataset, batch_size=1, shuffle=True)

    # load pre-trained network
    layer_shapes = []
    for _, arr in weights_dataset[0]['model_weights'].items():
        layer_shapes.append(arr.shape)
    net = load_network(options, layer_shapes)

    # bias metrics
    metrics_before = {'prediction': []}
    metrics_after = {'prediction': []}
    for metric in options['bias_metrics']:
        metrics_before[metric] = []
        metrics_after[metric] = []
    count = 0
    for data in weights_dataloader:
        count += 1
        print(f'Sample {count} / {options["num_samples"]}')
        if count > options['num_samples']:
            break

        # test with unbiased coloredMNIST dataset before bias mitigation
        digit_net = get_digit_classifier(data['model_weights'])
        for metric in options['bias_metrics']:
            if metric == "unbiased_accuracy":
                accuracy = accuracy_on_unbiased_mnist(digit_net)
                metrics_before[metric].append(accuracy)
            else:
                raise ValueError("Bias metric [{}] not recognized.".format(metric))

        if options['method']['name'] == "fast gradient sign method":
            new_data, prediction_before, prediction_after = fast_gradient_sign_method(net, data, epsilon=options['method']['epsilon'], steps=options['method']['steps'])
        else:
            raise ValueError("Bias mitigation method [{}] not recognized.".format(options['method']['name']))
        metrics_before['prediction'].append(prediction_before)
        metrics_after['prediction'].append(prediction_after)

        # test with unbiased coloredMNIST dataset after bias mitigation
        digit_net = get_digit_classifier(new_data['model_weights'])
        for metric in options['bias_metrics']:
            if metric == "unbiased_accuracy":
                accuracy = accuracy_on_unbiased_mnist(digit_net)
                metrics_after[metric].append(accuracy)
            else:
                raise ValueError("Bias metric [{}] not recognized.".format(metric))

    iter = 1
    for pb, ab, pa, aa in zip(metrics_before['prediction'], metrics_before['unbiased_accuracy'], metrics_after['prediction'], metrics_after['unbiased_accuracy']):
        print(f'Sample {iter}:')
        print(f'Accuracy: {ab:.5} ---> {aa:.5}')
        print(f'Prediction: {list(np.around(pb, decimals=5))} ---> {list(np.around(pa, decimals=5))}')

    plotMetrics(metrics_before,metrics_after)

def plotMetrics(metrics_before,metrics_after):
    """
    plot accuracy before mitigation and after mitigation, plots also mean prediction before and after mitigation
    """
    assert(metrics_before.keys() == metrics_after.keys())

    # plot predictions change
    pred_before_l = metrics_before['prediction']
    pred_after_l = metrics_after['prediction']
    pred_before = np.reshape(pred_before_l[0], (1,len(pred_before_l[0])))
    pred_after = np.reshape(pred_after_l[0], (1,len(pred_after_l[0])))
    for i in range(1,len(pred_before_l)):
        pred_before = np.append(pred_before, np.reshape(pred_before_l[i], (1,len(pred_before_l[i]))), axis=0)
        pred_after = np.append(pred_after, np.reshape(pred_after_l[i], (1,len(pred_after_l[i]))), axis=0)
    predictions = pd.DataFrame()
    classes = ['0.02','0.03','0.04','0.05']
    for i in range(0,len(classes)):
        predictions = predictions.append(
            {
                'category':classes[i],
                'before': np.mean(pred_before[:,i]),
                'after':np.mean(pred_after[:,i])
            }, ignore_index=True
        )
    predictions = predictions.set_index('category')
    fig = predictions.plot(kind="bar", title="prediction mean before and after mitigation", figsize=(10,10))
    fig.get_figure().savefig("plots/predictionAfterMitigation.jpg")

    # plot accuracy change 
    accuracy = pd.DataFrame()
    accuracy = accuracy.append({
            'accuracy': "before",
            'mean': np.mean(metrics_before["unbiased_accuracy"])
            },
            ignore_index=True)
    accuracy = accuracy.append({
            'accuracy': "after",
            'mean': np.mean(metrics_after['unbiased_accuracy'])
            },
            ignore_index=True)
    accuracy = accuracy.set_index('accuracy')
    fig = accuracy.plot(kind="bar", title="accuracy mean before and after mitigation", figsize=(10,10))
    fig.get_figure().savefig("plots/accuracyAfterMitigation.jpg")


if __name__ == '__main__':
    main()

