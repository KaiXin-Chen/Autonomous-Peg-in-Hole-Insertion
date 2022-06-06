"""
This file will include the networks for imitation learning
"""
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn

class Imi_networks(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 9),
        )

    def forward(self, x):
        return self.model(x)



class rnn_imi_networks(nn.Module):

    def __init__(self, n_features=1024,hidden_size=512,seq_len=10,num_layers=2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 9)

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        lstm_out=lstm_out.view(-1, self.hidden_size)
        y_pred = self.linear(lstm_out[:,-1,:]).view( -1, 1, 9)
        return y_pred


