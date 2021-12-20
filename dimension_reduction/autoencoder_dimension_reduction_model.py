import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


class AutoencoderDimensionReductionModel:
    def __init__(self, in_dim, out_dim=10, epochs=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)
        self.model = Autoencoder(in_dim, out_dim).to(self.device)
        self.loss_function_decoder = torch.nn.MSELoss()
        self.loss_function_classifier = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = epochs
        self.batch_size = 32
        self.loss_seq_decoder = []
        self.loss_seq_classifier = []

    def fit(self, data: pd.DataFrame, labels: pd.DataFrame):
        data = torch.tensor(data.values.astype(np.float32))
        labels = torch.tensor(labels.values.astype(np.int))
        dataset = torch.utils.data.TensorDataset(data, labels)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_seq_decoder = []
        self.loss_seq_classifier = []
        for epoch in range(self.epochs):
            train_loss_decoder = 0
            train_loss_classifier = 0
            for X, y in data_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                if epoch % 4 == 0:
                    self.model.encoder.requires_grad_(True)
                    self.model.decoder.requires_grad_(True)
                    self.model.classifier.requires_grad_(False)
                elif epoch % 4 == 1:
                    self.model.encoder.requires_grad_(False)
                    self.model.decoder.requires_grad_(False)
                    self.model.classifier.requires_grad_(True)
                elif epoch % 4 == 2:
                    self.model.encoder.requires_grad_(True)
                    self.model.decoder.requires_grad_(False)
                    self.model.classifier.requires_grad_(True)
                else:
                    self.model.encoder.requires_grad_(False)
                    self.model.decoder.requires_grad_(True)
                    self.model.classifier.requires_grad_(False)
                pred_X, pred_y = self.model(X)
                loss_decoder = self.loss_function_decoder(pred_X, X)
                loss_classifier = self.loss_function_classifier(pred_y, y[:,0])
                if epoch % 4 == 0 or epoch % 4 == 3:
                    loss = loss_decoder
                else:
                    loss = loss_classifier
                # else:
                #     loss = loss_decoder + loss_classifier
                loss.backward()
                self.optimizer.step()
                train_loss_decoder += loss_decoder.item() * X.size(0)
                train_loss_classifier += loss_classifier.item() * y.size(0)

            train_loss_decoder /= len(data_loader)
            train_loss_classifier /= len(data_loader)
            self.loss_seq_decoder.append(train_loss_decoder)
            self.loss_seq_classifier.append(train_loss_classifier)
            print(f"Epoch: {epoch+1} \tLoss Decoder: {train_loss_decoder:.6f} \tLoss Classifier: {train_loss_classifier:.6f}")

        plt.plot(self.loss_seq_decoder, label='decoder')
        plt.plot(np.array(self.loss_seq_classifier)/10, label='classifier')
        plt.ylabel("MSE/Cross Entropy Loss")
        plt.xlabel("epochs")
        plt.legend()
        plt.title("Training Loss")
        plt.show()

    def transform(self, data: pd.DataFrame):
        with torch.no_grad():
            data = torch.tensor(data.values.astype(np.float32)).to(self.device)
            return pd.DataFrame(self.model.encode(data).cpu().detach().numpy())

    def fit_transform(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.fit(data, labels)
        return self.transform(data)

    def save_model(self, name='autoencoder', opt=False):
        if not os.path.exists('data/models'):
            os.mkdir('data/models')

        plt.plot(self.loss_seq_decoder, label='decoder')
        plt.plot(np.array(self.loss_seq_classifier)/10, label='classifier')
        plt.ylabel("MSE/Cross Entropy Loss")
        plt.xlabel("epochs")
        plt.legend()
        plt.title("Training Loss")
        plt.savefig(f'data/results/{name}_loss.png')

        torch.save(self.model.state_dict(), f'data/models/{name}')
        if opt:
            torch.save(self.optimizer.state_dict(), f'data/models/{name}_optimizer')

    def load_model(self, name='autoencoder', opt=False):
        self.model.load_state_dict(torch.load(f'data/models/{name}'))
        if opt:
            self.optimizer.load_state_dict(torch.load(f'data/models/{name}_optimizer'))


class Autoencoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, factor=4):
        super(Autoencoder, self).__init__()

        first_dim = 1024
        while in_dim < first_dim:
            first_dim /= 2

        sizes = [in_dim]
        i = 0
        while first_dim / pow(factor, i) - first_dim / pow(factor, i) / 4 > out_dim:
            sizes.append(int(first_dim / pow(factor, i)))
            i += 1
        sizes.append(out_dim)

        encoder_layers = []
        for i in range(0, len(sizes)-1):
            encoder_layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            encoder_layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
            encoder_layers.append(torch.nn.ReLU(True))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(sizes)-1, 0, -1):
            decoder_layers.append(torch.nn.Linear(sizes[i], sizes[i - 1]))
            if i != 1:
                decoder_layers.append(torch.nn.BatchNorm1d(sizes[i - 1]))
                decoder_layers.append(torch.nn.ReLU(True))

        self.decoder = torch.nn.Sequential(*decoder_layers)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 4),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        x = self.decoder(encoded_x)
        y = self.classifier(encoded_x)
        return x, y

    def encode(self, x):
        return self.encoder(x)

    def classify(self, x):
        return self.classifier(x)
