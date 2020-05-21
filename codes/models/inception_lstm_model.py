import torch
import torch.nn.functional as F
import numpy as np

torch.device('cpu')


class InceptionBlock(torch.nn.Module):

    def __init__(self, in_channels, epsilon=0.001):
        super().__init__()

        def pad_same(f):
            pad = np.ceil((f - 1) / 2)
            return int(pad)

        self.branch1_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=64, stride=1)
        self.branch1_bn1 = torch.nn.BatchNorm1d(num_features=64, eps=epsilon)

        self.branch2_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=96, stride=1)
        self.branch2_bn1 = torch.nn.BatchNorm1d(num_features=96, eps=epsilon)
        self.branch2_conv3x3 = torch.nn.Conv1d(in_channels=96, kernel_size=3, out_channels=128, stride=1, padding=pad_same(3))
        self.branch2_bn2 = torch.nn.BatchNorm1d(num_features=128, eps=epsilon)

        self.branch3_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=16, stride=1)
        self.branch3_bn1 = torch.nn.BatchNorm1d(16, eps=epsilon)
        self.branch3_conv5x5 = torch.nn.Conv1d(in_channels=16, kernel_size=5, out_channels=32, stride=1, padding=pad_same(5))
        self.branch3_bn2 = torch.nn.BatchNorm1d(num_features=32, eps=epsilon)

        self.branch4_maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=pad_same(3))
        self.branch4_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=32, stride=1)
        self.branch4_bn1 = torch.nn.BatchNorm1d(num_features=32, eps=epsilon)

    def forward(self, X):

        # forward branch 1:
        x_branch1 = self.branch1_conv1x1(X)
        x_branch1 = self.branch1_bn1(x_branch1)
        x_branch1 = F.relu(x_branch1)

        # forward branch 2:
        x_branch2 = self.branch2_conv1x1(X)
        x_branch2 = self.branch2_bn1(x_branch2)
        x_branch2 = F.relu(x_branch2)
        x_branch2 = self.branch2_conv3x3(x_branch2)
        x_branch2 = self.branch2_bn2(x_branch2)
        x_branch2 = F.relu(x_branch2)

        # forward branch 3:
        x_branch3 = self.branch3_conv1x1(X)
        x_branch3 = self.branch3_bn1(x_branch3)
        x_branch3 = F.relu(x_branch3)
        x_branch3 = self.branch3_conv5x5(x_branch3)
        x_branch3 = self.branch3_bn2(x_branch3)
        x_branch3 = F.relu(x_branch3)

        # forward branch 4:
        x_branch4 = self.branch4_maxpool(X)
        x_branch4 = self.branch4_conv1x1(x_branch4)  # applying BN *after* activation
        x_branch4 = self.branch4_bn1(x_branch4)
        x_branch4 = F.relu(x_branch4)

        return torch.cat([x_branch1, x_branch2, x_branch3, x_branch4], dim=1)

# define the model:
class Inception_LSTM_model(torch.nn.Module):

    def __init__(self, n_features, hidden_dim, n_layers, label_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # init inception blocks:
        self.inception1 = InceptionBlock(in_channels=n_features)
        self.inception2 = InceptionBlock(in_channels=256)

        self.lstm = torch.nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)  # if batch_first=True: rnn input shape = (batch_size, seq_len, features), rnn output shape = (batch_size, seq_len, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, label_dim)

    def init_hidden(self, batch_size):
        # generate the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def forward(self, X):
        # X.shape = (batch_size, seq_len, features)
        batch_size = X.size(0)

        # change axis order for CNN:
        X = X.transpose(1, 2)

        # inception blocks:
        X = self.inception1(X)
        X = self.inception2(X)

        # change axis order for LSTM:
        X = X.transpose(1, 2)

        # LSTM step:
        hidden_state = self.init_hidden(batch_size)
        cell_state = self.init_hidden(batch_size)
        X, _ = self.lstm(X, (hidden_state, cell_state)) # out shape = (batch_size, seq_len, hidden_size)
        X = X[:, -1, :]

        # fc step:
        X = self.bn2(X)
        X = self.fc(X)
        X = torch.sigmoid(X)

        return X.view(-1, )
