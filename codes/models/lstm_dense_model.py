import torch
import torch.nn.functional as F

torch.device('cpu')


# define the model:
class LSTM_DENSE_model(torch.nn.Module):

    def __init__(self, n_features, hidden_dim, n_layers, fc_channels, n_labels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.fc_channel1, self.fc_channel2 = fc_channels

        self.lstm = torch.nn.LSTM(n_features, self.hidden_dim, self.n_layers, batch_first=True)  # batch_first=True: rnn input shape = (batch_size, seq_len, features), rnn output shape = (batch_size, seq_len, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, self.fc_channel1)
        #self.dropout1 = torch.nn.Dropout(p)
        self.fc2 = torch.nn.Linear(self.fc_channel1, self.fc_channel2)
        #self.dropout2 = torch.nn.Dropout(p)
        self.fc3 = torch.nn.Linear(self.fc_channel2, n_labels)


    def init_hidden(self, batch_size):
        # generate the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def forward(self, X):
        # X: batch data, tensor type, shape = (batch_size, seq_len, features)
        batch_size = X.size(0)


        # lstm step
        hidden_state = self.init_hidden(batch_size)
        cell_state = self.init_hidden(batch_size)
        X, _ = self.lstm(X, (hidden_state, cell_state)) # out shape = (batch_size, seq_len, hidden_size)
        X = X[:, -1, :]

        # fc step
        X = self.bn1(X)

        X = self.fc1(X)
        #X = self.dropout1(X)
        X = F.relu(X)
        X = self.fc2(X)
        #X = self.dropout2(X)
        X = F.relu(X)
        X = self.fc3(X)



        X = torch.sigmoid(X)


        return X.view(-1, )
