import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, output_size=100):
        super(LSTMNetwork, self).__init__()

        self.output_size = output_size

        self.lstm = nn.LSTM(3, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 2 * self.output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        
        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.relu(self.fc(x))

        return x.view(-1, self.output_size, 2) 