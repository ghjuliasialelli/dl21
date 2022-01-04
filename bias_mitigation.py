import math
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
from torch.nn.init import kaiming_uniform_
import time
import pickle


# DATASETS

def load_network_weights_dataset():
    from utils.model_operations import LucasModelDataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return LucasModelDataset(device, ['data/digitWdb_train.pkl.gz', 'data/digitWdb_test.pkl.gz'], only_conv=False)


def load_unbiased_color_mnist_dataset(train=False):
    from utils.image_operations import LucasDigitData
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not train:
        return LucasDigitData(device, [
            'data/colored_mnist/mnist_10color_jitter_var_0.050.npy'
        ], train=False, test=True)
    else:
        return LucasDigitData(device, [
            'data/colored_mnist/mnist_10color_jitter_var_0.045.npy'
        ], train=False, test=True)


# NETWORKS

def load_network(options, layer_shapes):
    # create network
    if options['network'] == 'conv2d_ifbid':
        from models.ifbid import Conv2D_IFBID_Model
        net = Conv2D_IFBID_Model(layer_shapes=layer_shapes, use_dense=True, num_classes=4, batch_size=1)
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
    # start_time = time.time()
    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset()
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=64)

    net.eval()

    with torch.no_grad():
        accuracy = 0
        for X, y in mnist_dataloader:
            pred = np.argmax(net(X).detach().cpu().numpy(), axis=1)
            true = y[:, 0].detach().cpu().numpy()
            accuracy += np.count_nonzero(pred == true)

        accuracy /= (len(mnist_dataloader) * 64)

    # print(f"Accuracy time: {time.time() - start_time:.3f}s")
    return accuracy


def bias_score(net):
    # start_time = time.time()
    meanColors = np.array([
        [220, 20, 60],
        [0, 128, 128],
        [253, 233, 16],
        [0, 149, 182],
        [237, 145, 33],
        [145, 30, 188],
        [70, 240, 240],
        [250, 197, 187],
        [210, 245, 60],
        [128, 0, 0]
    ])

    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset()
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=64)

    net.eval()

    with torch.no_grad():
        score = 0
        count = 0
        for X, y in mnist_dataloader:
            true = y[:, 0].detach().cpu().numpy()
            results = np.empty((len(meanColors), len(y)))
            for i in range(len(meanColors)):
                preds = net(change_mnist_color(meanColors[i], X)).detach().cpu().numpy()
                mask = np.logical_or(np.equal(true, np.argmax(preds, axis=1)), true != i)
                results = results[:, mask]
                preds = preds[mask]
                true = true[mask]
                X = X[mask]
                results[i] = preds[[l for l in range(len(true))], true]

            results /= np.expand_dims(results[true, [l for l in range(len(true))]], axis=0)
            score += np.sum(results) - results.shape[1]
            count += (results.shape[0] - 1) * results.shape[1]

    # print(f"Bias Score time: {time.time() - start_time:.3f}s")

    return score / count


def change_mnist_color(color, image):
    color_map = torch.logical_or(torch.logical_or(image[:, 0] > 0, image[:, 1] > 0), image[:, 2] > 0)
    new_color = image.clone()
    for i in range(3):
        new_color[:, i][color_map] = color[i]
    return new_color


# LOCALIZATION/MITIGATION METHODS

def fast_gradient_sign_method(net, data, epsilon=2e-2, steps=1):
    # start_time = time.time()
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fcn = torch.nn.CrossEntropyLoss()

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        if k.startswith('bias'):
            new_weights[k] = v.clone()
        else:
            new_weights[k] = v.clone().requires_grad_()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    for param in net.parameters():
        param.requires_grad = False

    # FGSM steps
    for i in range(steps):
        pred = net(new_data['model_weights'])
        loss = loss_fcn(pred.unsqueeze(0), torch.tensor([3], device=device).long())
        loss.backward()

        for layer in new_data['model_weights']:
            if layer.startswith("bias"):
                continue
            sign_grad = torch.sign(new_data['model_weights'][layer].grad.to(device))
            with torch.no_grad():
                new_data['model_weights'][layer] -= epsilon * sign_grad
            new_data['model_weights'][layer].grad.data.zero_()

    # no grad on weights
    for layer in new_data['model_weights']:
        new_data['model_weights'][layer].requires_grad_(False)

    # create mnist classifier
    digit_net = get_digit_classifier(new_data['model_weights'])

    # prediction after
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()

    # print(f"FGSM time: {time.time() - start_time:.3f}s")

    return digit_net, new_data, prediction_after


def gradient_method(net, data, epsilon=2e-2, steps=1):
    # start_time = time.time()

    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fcn = torch.nn.CrossEntropyLoss()

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        if k.startswith('bias'):
            new_weights[k] = v.clone()
        else:
            new_weights[k] = v.clone().requires_grad_()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    for param in net.parameters():
        param.requires_grad = False

    # FGSM steps
    for i in range(steps):
        pred = net(new_data['model_weights'])
        loss = loss_fcn(pred.unsqueeze(0), torch.tensor([3], device=device).long())
        loss.backward()

        for layer in new_data['model_weights']:
            if layer.startswith("bias"):
                continue
            grad = new_data['model_weights'][layer].grad.clone().to(device)
            with torch.no_grad():
                new_data['model_weights'][layer] -= epsilon * grad / torch.max(torch.abs(grad))
            new_data['model_weights'][layer].grad.data.zero_()

    # no grad on weights
    for layer in new_data['model_weights']:
        new_data['model_weights'][layer].requires_grad_(False)

    # create mnist classifier
    digit_net = get_digit_classifier(new_data['model_weights'])

    # prediction after
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()

    # print(f"Gradient time: {time.time() - start_time:.3f}s")

    return digit_net, new_data, prediction_after


def weight_reset(net, data, layer=None, neurons=None):
    # start_time = time.time()

    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # copy data data
    new_weights = {}
    for k, v in data['model_weights'].items():
        new_weights[k] = v.clone()
    new_data = {'model_weights': new_weights, 'bias': data['bias'].clone()}

    for l in new_data['model_weights']:
        if l.startswith('bias'):
            continue
        if layer is None or l == layer:
            if neurons is not None:
                weight_reset = new_data['model_weights'][l].clone()
                kaiming_uniform_(weight_reset, a=math.sqrt(5))
                new_data['model_weights'][l] = new_data['model_weights'][l] * torch.logical_not(torch.from_numpy(neurons[l]).to(device)) \
                                                   + weight_reset * torch.from_numpy(neurons[l]).to(device)
            else:
                kaiming_uniform_(new_data['model_weights'][l], a=math.sqrt(5))

    # create mnist classifier
    digit_net = get_digit_classifier(new_data['model_weights'])

    # prediction before
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()

    # print(f"Weight Reset time: {time.time() - start_time:.3f}s")

    return digit_net, prediction_after


def dropout(net, data, prob=0.5, layer=None):
    # start_time = time.time()

    net.eval()

    # prediction before
    with torch.no_grad():
        prediction_after = net(data['model_weights']).detach().cpu().numpy()

    # create mnist classifier
    digit_net = get_digit_classifier(data['model_weights'])
    digit_net.add_dropout(prob, layer)

    # print(f"Dropout time: {time.time() - start_time:.3f}s")

    return digit_net, prediction_after


def fine_tune(net, data, epochs=1, lr=1e-4, layer=None):
    # start_time = time.time()

    net.eval()

    # create mnist classifier
    digit_net = get_digit_classifier(data['model_weights'])

    # load unbiased colored MNIST dataset
    mnist_dataset = load_unbiased_color_mnist_dataset(train=True)
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=8)

    digit_net.train()
    optimizer = torch.optim.Adam(digit_net.parameters(), lr=lr)
    loss_fcn = torch.nn.CrossEntropyLoss()
    for param in digit_net.parameters():
        param.requires_grad = False
    for name, lay in digit_net.named_children():
        if layer == "layer_0" and name == "conv1":
            for param in lay.parameters():
                param.requires_grad = True
        elif layer == "layer_1" and name == "conv2":
            for param in lay.parameters():
                param.requires_grad = True
        elif layer == "layer_2" and name == "conv3":
            for param in lay.parameters():
                param.requires_grad = True
        elif layer == "layer_3" and name == "dense1":
            for param in lay.parameters():
                param.requires_grad = True
        elif layer == "layer_4" and name == "dense2":
            for param in lay.parameters():
                param.requires_grad = True

    for epoch in range(epochs):
        for X, y in mnist_dataloader:
            optimizer.zero_grad()
            pred = digit_net.forward(X)
            loss = loss_fcn(pred, y[:, 0].long())
            loss.backward()
            optimizer.step()

    for param in digit_net.parameters():
        param.requires_grad = True

    new_data = {'model_weights': digit_net.get_model_weights(), 'bias': data['bias'].clone()}

    # prediction before
    with torch.no_grad():
        prediction_after = net(new_data['model_weights']).detach().cpu().numpy()

    # print(f"Fine-Tune time: {time.time() - start_time:.3f}s")

    return digit_net, prediction_after


# MAIN

def main(options):
    # set seed
    random.seed(options['random_seed'])
    np.random.seed(options['random_seed'])
    torch.manual_seed(options['random_seed'])

    # load network weights dataset
    weights_dataset = load_network_weights_dataset()
    weights_dataloader = DataLoader(weights_dataset, batch_size=1, shuffle=True)

    # load pre-trained network
    layer_shapes = []
    for layer, arr in weights_dataset[0]['model_weights'].items():
        if layer.startswith("layer"):
            layer_shapes.append(arr.shape)
    net = load_network(options, layer_shapes)

    # result values
    if options['start_num_samples'] == 0:
        result = {
            "prediction": [],
            "accuracy": [],
            "bias_score": [],
            "actual_bias": []
        }
        selected_layers = {}
        changesPerSample = {}
        for method in options['methods']:
            selected_layers[method['localization_method']['name']] = []
            changesPerSample[method['localization_method']['name']] = []
            for mitigation in method['mitigation_methods']:
                if mitigation is None:
                    result[f"{method['localization_method']['name']}"] = {
                        "prediction": [],
                        "accuracy": [],
                        "bias_score": []
                    }
                else:
                    result[f"{method['localization_method']['name']}_{mitigation['name']}"] = {
                        "prediction": [],
                        "accuracy": [],
                        "bias_score": []
                    }
    else:
        with open(f"plots/{options['name']}_saved_result.pkl", "rb") as f:
            result = pickle.load(f)
        with open(f"plots/{options['name']}_saved_changes_per_sample.pkl", "rb") as f:
            changesPerSample = pickle.load(f)
        with open(f"plots/{options['name']}_saved_selected_layers.pkl", "rb") as f:
            selected_layers = pickle.load(f)


    for method in options['methods']:
        if method['localization_method']['name'] == "permutation_importance":
            if options['start_num_samples'] == 0:
                # compute permutation importance by layer
                permutation_importance_by_layer = get_permutation_importance_by_layers(net, weights_dataloader, method['localization_method']['iterations'])
                with open(f"plots/{options['name']}_permutation_importance.pkl", "wb") as f:
                    pickle.dump(permutation_importance_by_layer, f)
            else:
                with open(f"plots/{options['name']}_permutation_importance.pkl", "rb") as f:
                    permutation_importance_by_layer = pickle.load(f)
            plotPermutationImportance(permutation_importance_by_layer)
            perm_imp_mitigation_layer = f"layer_{np.argmax(permutation_importance_by_layer['importances_mean'])}"
            break

    # bias metrics
    count = [0]*4
    for data in weights_dataloader:
        # same amount of samples from each bias level
        bias = np.argmax(data['bias'][0].detach().cpu().numpy())
        if count[bias] >= options['num_samples']:
            if np.sum(count) == 4 * options['num_samples']:
                break
            else:
                continue
        count[bias] += 1

        if count[bias] <= options['start_num_samples']:
            continue

        start_time = time.time()

        # save actual bias
        result["actual_bias"].append(bias)

        # test with unbiased coloredMNIST dataset before bias mitigation
        digit_net = get_digit_classifier(data['model_weights'])
        accuracy = accuracy_on_unbiased_mnist(digit_net)
        result["accuracy"].append(accuracy)
        score = bias_score(digit_net)
        result["bias_score"].append(score)

        # prediction before
        with torch.no_grad():
            result['prediction'].append(net(data['model_weights']).detach().cpu().numpy())

        for method in options['methods']:
            # ablation study (& bias mitigation)
            if method['localization_method']['name'] != "permutation_importance":
                if method['localization_method']['name'] == "fgsm":
                    digit_net, new_data, prediction_after = fast_gradient_sign_method(net, data, epsilon=method['localization_method']['epsilon'], steps=method['localization_method']['steps'])
                elif method['localization_method']['name'] == "gradient":
                    digit_net, new_data, prediction_after = gradient_method(net, data, epsilon=method['localization_method']['epsilon'], steps=method['localization_method']['steps'])
                else:
                    raise ValueError("Bias localization/mitigation method [{}] not recognized.".format(method['localization_method']['name']))

                # TODO plot ablation study and get select layer or neurons
                changesPerLayer = computeAbsoluteChangesInSample(data, new_data)
                mitigation_neurons = {}

                for layer in changesPerLayer:
                    mitigation_neurons[layer] = changesPerLayer[layer] >= method['localization_method']['neuron_threshold']
                    changesPerLayer[layer] = np.mean(changesPerLayer[layer])
                changesPerSample[method['localization_method']['name']].append(changesPerLayer)
                mitigation_layer = get_most_biased_layer(changesPerLayer)
            else:
                mitigation_layer = perm_imp_mitigation_layer
                mitigation_neurons = {}
            selected_layers[method['localization_method']['name']].append(mitigation_layer)

            for mitigation in method['mitigation_methods']:
                if mitigation is None:
                    name = method['localization_method']['name']
                else:
                    name = f"{method['localization_method']['name']}_{mitigation['name']}"

                    # bias mitigation
                    if mitigation['name'] == "layer_reset":
                        digit_net, prediction_after = weight_reset(net, data, layer=mitigation_layer)
                    elif mitigation['name'] == "neuron_reset":
                        digit_net, prediction_after = weight_reset(net, data, neurons=mitigation_neurons)
                    elif mitigation['name'] == "dropout":
                        digit_net, prediction_after = dropout(net, data, prob=mitigation['prob'], layer=mitigation_layer)
                    elif mitigation['name'] == "fine_tune":
                        digit_net, prediction_after = fine_tune(net, data, epochs=mitigation['epochs'], lr=mitigation['lr'], layer=mitigation_layer)
                    else:
                        raise ValueError("Bias mitigation method [{}] not recognized.".format(mitigation['name']))

                # test with unbiased coloredMNIST dataset after bias mitigation
                result[name]["prediction"].append(prediction_after)
                accuracy = accuracy_on_unbiased_mnist(digit_net)
                result[name]["accuracy"].append(accuracy)
                score = bias_score(digit_net)
                result[name]["bias_score"].append(score)

        print(f'Sample {np.sum(count)} / {options["num_samples"] * 4}, Time: {time.time() - start_time:.3f}s')

        # save every 10 samples
        if np.sum(count) % 10 == 0:
            with open(f"plots/{options['name']}_saved_result.pkl", "wb") as f:
                pickle.dump(result, f)
            with open(f"plots/{options['name']}_saved_changes_per_sample.pkl", "wb") as f:
                pickle.dump(changesPerSample, f)
            with open(f"plots/{options['name']}_saved_selected_layers.pkl", "wb") as f:
                pickle.dump(selected_layers, f)

    with open(f"plots/{options['name']}_saved_result.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(f"plots/{options['name']}_saved_changes_per_sample.pkl", "wb") as f:
        pickle.dump(changesPerSample, f)
    with open(f"plots/{options['name']}_saved_selected_layers.pkl", "wb") as f:
        pickle.dump(selected_layers, f)

    visualizeChanges(changesPerSample, selected_layers)
    plotMetrics(result, options)


def plotMetrics(result, options):
    """
    plot accuracy before mitigation and after mitigation, plots also mean prediction before and after mitigation
    """

    for method in options['methods']:
        for mitigation in method['mitigation_methods']:
            if mitigation is None:
                name = method['localization_method']['name']
            else:
                name = f"{method['localization_method']['name']}_{mitigation['name']}"

            # plot predictions change
            pred_before_l = result['prediction']
            pred_after_l = result[name]['prediction']
            pred_before = np.reshape(pred_before_l[0], (1,len(pred_before_l[0])))
            pred_after = np.reshape(pred_after_l[0], (1,len(pred_after_l[0])))
            for i in range(1, len(pred_before_l)):
                pred_before = np.append(pred_before, np.reshape(pred_before_l[i], (1,len(pred_before_l[i]))), axis=0)
                pred_after = np.append(pred_after, np.reshape(pred_after_l[i], (1,len(pred_after_l[i]))), axis=0)
            predictions = pd.DataFrame()
            classes = ['0.02','0.03','0.04','0.05']
            for i in range(0,len(classes)):
                predictions = predictions.append(
                    {
                        'color jitter variance': classes[i],
                        'before': np.mean(pred_before[:,i]),
                        'after': np.mean(pred_after[:,i])
                    }, ignore_index=True
                )
            predictions = predictions.set_index('color jitter variance')
            fig = predictions.plot(kind="bar", title="Prediction Before and After Mitigation", figsize=(10,10), ylabel="mean prediction score")
            fig.get_figure().savefig(f"plots/{name}_bias_predictions.png")
            plt.close(fig.get_figure())

            # plot accuracy change
            biases = ['0.02', '0.03', '0.04', '0.05']
            actual_bias = np.array(result['actual_bias'])
            accuracy = pd.DataFrame()
            for i, b in enumerate(biases):
                accuracy = accuracy.append({
                        'color jitter variance': b,
                        'before': np.mean(np.array(result["accuracy"])[np.array(actual_bias) == i]),
                        'after': np.mean(np.array(result[name]['accuracy'])[np.array(actual_bias) == i])
                        },
                        ignore_index=True)
            accuracy = accuracy.set_index('color jitter variance')
            fig = accuracy.plot(kind="bar", title="Accuracy on Unbiased Dataset Before and After Mitigation", figsize=(10, 10), ylabel="mean accuracy")
            fig.get_figure().savefig(f"plots/{name}_accuracy.jpg")
            plt.close(fig.get_figure())

            # plot bias score change
            biases = ['0.02', '0.03', '0.04', '0.05']
            actual_bias = np.array(result['actual_bias'])
            accuracy = pd.DataFrame()
            for i, b in enumerate(biases):
                accuracy = accuracy.append({
                    'color jitter variance': b,
                    'before': np.mean(np.array(result["bias_score"])[np.array(actual_bias) == i]),
                    'after': np.mean(np.array(result[name]['bias_score'])[np.array(actual_bias) == i])
                },
                    ignore_index=True)
            accuracy = accuracy.set_index('color jitter variance')
            fig = accuracy.plot(kind="bar", title="Bias Score Before and After Mitigation",
                                figsize=(10, 10), ylabel="mean bias score")
            fig.get_figure().savefig(f"plots/{name}_bias_score.jpg")
            plt.close(fig.get_figure())


def get_most_biased_layer(changesPerLayer):
    mean_changes = []
    for layer in changesPerLayer:
        mean_changes.append((layer, changesPerLayer[layer]))
    mean_changes.sort(key=lambda x: x[1])
    return mean_changes[-1][0]


def computeAbsoluteChangesInSample(data, new_data):
    """
    computes absolut weight change per neuron for every sample
    """
    changesPerLayer = {}
    for layer in data['model_weights']:
        if layer.startswith("bias"):
            continue
        assert (np.shape(data['model_weights'][layer]) == np.shape(new_data['model_weights'][layer]))
        absoluteChange = np.abs(data['model_weights'][layer].detach().cpu().numpy() - new_data['model_weights'][layer].detach().cpu().numpy())
        changesPerLayer[layer] = absoluteChange
    return changesPerLayer


def visualizeChanges(changesPerSample, selected_layers):
    """
    plots mean of mean change per sampled layer and mean of mean change per sampled neuron (both on layer level)
    """
    for name in changesPerSample:
        if changesPerSample[name] == []:
            continue

        # transform data to plot
        dictionary = {}
        sample = changesPerSample[name][0]
        for layer in sample:
            dictionary[layer] = []

        for sample in changesPerSample[name]:
            for layer in sample:
                dictionary[layer].append(sample[layer])

        results = pd.DataFrame()
        for layer in dictionary.keys():
            results = results.append({
                'mean': np.mean(dictionary[layer]),
                'std': np.std(dictionary[layer])
            }, ignore_index=True)
        results.index = ['conv1', 'conv2', 'conv3', 'dense1', 'dense2']
        fig = results.plot(kind='bar', figsize=(10, 10), title="Overall Changes", ylabel="mean difference of weights", xlabel="layer")
        fig.get_figure().savefig(f"plots/{name}_overall_changes.png")
        plt.close(fig.get_figure())


    for name in selected_layers:
        if selected_layers[name] == []:
            continue

        results = pd.DataFrame()
        for layer in ['layer_0', 'layer_1', 'layer_2', 'layer_3', 'layer_4']:
            results = results.append({
                'count': np.mean(np.sum(np.array(selected_layers[name]) == layer))
            }, ignore_index=True)
        results.index = ['conv1', 'conv2', 'conv3', 'dense1', 'dense2']
        fig = results.plot(kind='bar', figsize=(10, 10), title="Most Biased Layer", ylabel="count", xlabel="layer")
        fig.get_figure().savefig(f"plots/{name}_most_biased_layer.png")
        plt.close(fig.get_figure())


def get_permutation_importance_by_layers(net, weights_dataloader, number_of_iterations):
    start = time.time()

    # compute reference score s (accuracy of the net when evaluated on the whole training data)
    net.eval()
    with torch.no_grad():
        trues = []
        preds = []
        for data in weights_dataloader:
            preds.append(np.argmax(net(data['model_weights']).detach().cpu().numpy()))
            trues.append(np.argmax(data['bias'].detach().cpu().numpy()))
        preds = np.array(preds)
        trues = np.array(trues)
    s = np.count_nonzero(preds == trues) / len(preds)
    print(s)

    end = time.time()
    print("reference evaluation done", end-start)
    start = time.time()
    
    
    # this code is only of the initialization of the data frame
    weights_size = 0
    layerNames = list(weights_dataloader.dataset[0]['model_weights'].keys())
    layer_numel = {}
    for layer in weights_dataloader.dataset[0]['model_weights']:
        if layer.startswith("bias"):
            continue
        layer_numel[layer] = np.prod(weights_dataloader.dataset[0]['model_weights'][layer].detach().cpu().numpy().shape)
        weights_size += layer_numel[layer]
    df = np.empty((len(weights_dataloader), weights_size), dtype='float32')

    end = time.time()
    print("data frame initialization done", end-start)
    start = time.time()


    # fills the data frame
    for i, data in enumerate(weights_dataloader):
        layerValues = []
        for layer in data['model_weights'].keys():
            if layer.startswith("bias"):
                continue
            layerValues.extend(list(np.ravel(data['model_weights'][layer].detach().cpu().numpy())))
        df[i] = np.array(layerValues)

        if (i+1) % 100 == 0:
            print(f"{i + 1} / {len(weights_dataloader)}, time: {time.time() - start:.5f}s")


    end = time.time()
    print("data frame filling done", end-start)
    start = time.time()
    results = {
        'importances_mean': [],
        'importances_std': []
    }

    # we permutate layer by layer
    start_col = 0
    for ln in layerNames:
        if ln.startswith("bias"):
            continue
        # we only work on a copy of the original data frame and reset it when permutating a new layer
        dfCopy = df.copy()
        repeats = []
        layer_weights_numel = layer_numel[ln]

        # repeat process for each feature number_of_iterations times
        for k in range(number_of_iterations):
            # permutate all columns associated to a layer
            for col in range(start_col, start_col + layer_weights_numel):
                dfCopy[:, col] = np.random.permutation(dfCopy[:, col])

            # compute s_k_j which is equivalent to the accuracy measured on the data from dfCopy
            net.eval()
            with torch.no_grad():
                trues = []
                preds = []
                for idx, data in enumerate(weights_dataloader):
                    layer_weights = np.reshape(dfCopy[idx, start_col:start_col+layer_weights_numel], data['model_weights'][ln].shape)
                    data['model_weights'][ln] = torch.from_numpy(layer_weights).to(data['model_weights'][ln].device)
                    preds.append(np.argmax(net(data['model_weights']).detach().cpu().numpy()[0]))
                    trues.append(np.argmax(data['bias'].detach().cpu().numpy()[0]))
                preds = np.array(preds)
                trues = np.array(trues)

            s_k_j = np.count_nonzero(preds == trues) / len(preds)

            print(s_k_j)

            repeats.append(s_k_j)

        end = time.time()
        print("finished", ln, end-start)
        start = time.time()

        importances = list(np.array([s]*len(repeats))-np.array(repeats))
        # calculate importance mean
        results['importances_mean'].append(np.mean(importances))
        # calculate importance standard deviation
        results['importances_std'].append(np.std(importances))

        start_col += layer_weights_numel

    return results

def plotPermutationImportance(resultDict):
    print(resultDict)
    df = pd.DataFrame(resultDict)

    df.columns = ['importance mean', 'importance std']
    df.index = ['conv1','conv2','conv3','dense1','dense2']
    fig = df.plot(kind='bar', figsize=(10,10), title="Permutation Importance", ylabel="decrease in accuracy", xlabel="layer")
    fig.get_figure().savefig("plots/permutation_importance.png")
    plt.close(fig.get_figure())


def plotResults(options):
    with open(f"plots/{options['name']}_saved_result.pkl", "rb") as f:
        result = pickle.load(f)
    with open(f"plots/{options['name']}_saved_changes_per_sample.pkl", "rb") as f:
        changesPerSample = pickle.load(f)
    with open(f"plots/{options['name']}_saved_selected_layers.pkl", "rb") as f:
        selected_layers = pickle.load(f)

    name_to_title = {
        "before": "Before Mitigation",
        "gradient": "Gradient Method",
        "gradient_layer_reset": "Reset Weights (Layer-Wise)",
        "gradient_neuron_reset": "Reset Weights (Neuron-Wise)",
        "gradient_dropout": "Dropout",
        "gradient_fine_tune": "Fine-Tune",
        "fgsm": "Fast Gradient Sign Method",
        "fgsm_layer_reset": "Reset Weights (Layer-Wise)",
        "fgsm_neuron_reset": "Reset Weights (Neuron-Wise)",
        "fgsm_dropout": "Dropout",
        "fgsm_fine_tune": "Fine-Tune"
    }

    visualizeChanges(changesPerSample, selected_layers)

    # plot predictions change
    preds = {}
    preds['before'] = np.array(result['prediction'])
    for method in options['methods']:
        for mitigation in method['mitigation_methods']:
            if mitigation is None:
                name = method['localization_method']['name']
            else:
                name = f"{method['localization_method']['name']}_{mitigation['name']}"
            preds[name] = np.array(result[name]['prediction'])
    predictions = pd.DataFrame()
    classes = ['0.02','0.03','0.04','0.05']
    for i in range(0,len(classes)):
        temp = {'color jitter variance': classes[i]}
        for k in preds:
            temp[name_to_title[k]] = np.mean(preds[k][:, i])
        predictions = predictions.append(temp, ignore_index=True)
    predictions = predictions.set_index('color jitter variance')
    fig = predictions.plot(kind="bar", title="Prediction Before and After Mitigation", figsize=(10,10), ylabel="mean prediction score")
    fig.get_figure().savefig(f"plots/all_bias_predictions.png")
    plt.close(fig.get_figure())

    # plot accuracy change
    biases = ['0.02', '0.03', '0.04', '0.05']
    actual_bias = np.array(result['actual_bias'])
    accuracy = pd.DataFrame()
    for i, b in enumerate(biases):
        temp = {'color jitter variance': b, 'before': np.mean(np.array(result["accuracy"])[np.array(actual_bias) == i])}
        for method in options['methods']:
            for mitigation in method['mitigation_methods']:
                if mitigation is None:
                    name = method['localization_method']['name']
                else:
                    name = f"{method['localization_method']['name']}_{mitigation['name']}"
                temp[name_to_title[name]] = np.mean(np.array(result[name]['accuracy'])[np.array(actual_bias) == i])

        accuracy = accuracy.append(temp, ignore_index=True)
    accuracy = accuracy.set_index('color jitter variance')
    fig = accuracy.plot(kind="bar", title="Accuracy on Unbiased Dataset Before and After Mitigation", figsize=(10, 10), ylabel="mean accuracy")
    fig.get_figure().savefig(f"plots/all_accuracy.jpg")
    plt.close(fig.get_figure())

    # plot bias score change
    biases = ['0.02', '0.03', '0.04', '0.05']
    actual_bias = np.array(result['actual_bias'])
    accuracy = pd.DataFrame()
    for i, b in enumerate(biases):
        temp = {'color jitter variance': b, 'before': np.mean(np.array(result["bias_score"])[np.array(actual_bias) == i])}
        for method in options['methods']:
            for mitigation in method['mitigation_methods']:
                if mitigation is None:
                    name = method['localization_method']['name']
                else:
                    name = f"{method['localization_method']['name']}_{mitigation['name']}"
                temp[name_to_title[name]] = np.mean(np.array(result[name]['bias_score'])[np.array(actual_bias) == i])
        accuracy = accuracy.append(temp, ignore_index=True)
    accuracy = accuracy.set_index('color jitter variance')
    fig = accuracy.plot(kind="bar", title="Bias Score Before and After Mitigation",
                        figsize=(10, 10), ylabel="mean bias score")
    fig.get_figure().savefig(f"plots/all_bias_score.jpg")
    plt.close(fig.get_figure())

if __name__ == '__main__':
    # load options
    with open('options/bias_mitigation_options4.json', 'r') as f:
        options = json.load(f)
    main(options)
    # plotResults(options)
