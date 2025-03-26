import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, output_size=1, hidden_size=256):
        super(LSTMNetwork, self).__init__()

        self.output_size = output_size

        self.lstm = nn.LSTM(2, hidden_size, 4, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * self.output_size)

        self.relu = nn.LeakyReLU()

    def __forward_once(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return torch.abs(x).view(-1, self.output_size, 2)

    def forward(self, x, time):
        batch_size = x.shape[0]

        output_seq = torch.zeros(batch_size, 0, 2, device=x.device)

        for _ in range(time):
            output = self.__forward_once(x)

            output_seq = torch.cat((output_seq, output), dim=1)

            x = torch.cat((x[:, 1:], output), dim=1) 

        return output_seq
