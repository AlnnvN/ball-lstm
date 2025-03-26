import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, output_size=100, hidden_size=256):
        super(LSTMNetwork, self).__init__()

        self.output_size = output_size

        self.lstm = nn.LSTM(3, hidden_size, 4, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * self.output_size)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        #return torch.abs(x).view(-1, self.output_size, 2).round(decimals=3)
        return torch.abs(x).view(-1, self.output_size, 2)
