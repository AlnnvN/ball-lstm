import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, output_size=1, hidden_size=32, training=False, using_velocity=True):
        super(LSTMNetwork, self).__init__()

        self.training = training

        self.output_size = output_size

        self.using_velocity = using_velocity

        feature_size = 4 if using_velocity else 2

        self.lstm = nn.LSTM(feature_size, hidden_size, 4, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, 2 * self.output_size)

        self.relu = nn.LeakyReLU()

    def forward_autoregressive(self, past, future_len=1, teacher_forcing_ratio=0.5, future_gt=None):
        _, (h, c) = self.lstm(past)

        input_t = past[:, -1, :2]
        outputs = torch.zeros(past.size(0), 0, 2, device=past.device)

        if future_gt is not None:
            future_len = future_gt.shape[1]

        for t in range(future_len):
            out_lstm, (h, c) = self.lstm(input_t.unsqueeze(1), (h, c))
            pred = self.fc2(self.relu(self.fc1(out_lstm.squeeze(1))))
            pred = pred.view(-1, 1, 2)  # apenas um passo por vez

            if outputs.shape[1] == 0:
                outputs = pred
            else:
                diff = pred - past[:, -1:, :]  # deslocamento
                outputs = torch.cat((outputs, outputs[:, -1:, :] + diff), dim=1)

            # decide teacher forcing
            if self.training and future_gt is not None and torch.rand(1) < teacher_forcing_ratio:
                next_input = future_gt[:, t, :].unsqueeze(1)
            else:
                next_input = pred

            # atualiza janela e normaliza
            past = torch.cat((past[:, 1:], next_input), dim=1)
            past = past - past[:, 0:1, :]
            input_t = next_input.squeeze(1)

        return outputs




    def forward(self, x, time):
        if self.using_velocity:
            return self.__complete_forward_with_velocity(x, time)
        else:
            return self.__complete_forward(x, time)

    def __complete_forward(self, x, time):
        batch_size = x.shape[0]
        
        output_seq = torch.zeros(batch_size, 0, 2, device=x.device)

        _, (h, c) = self.lstm(past)

        for _ in range(time):
            # print(f'x -> {x}')

            output = self.forward_once(x)

            # print(f'output -> {output}')

            if output_seq.shape[1] == 0:
                output_seq = output
            else:
                diff = output - x[:, -1, :]  
                output_seq = torch.cat((output_seq, output_seq[:, -1:, :] + diff), dim=1)

            # print(f'output_seq -> {output_seq}')

            x = torch.cat((x[:, 1:], output), dim=1)

            # print(f'x concat -> {x}')

            x = x - x[:, 0:1, :]

        return output_seq

    def __complete_forward_with_velocity(self, x, time):
        batch_size = x.shape[0]
        
        output_seq = torch.zeros(batch_size, 0, 2, device=x.device)

        for _ in range(time):
            # print(f'x -> {x}')

            output = self.forward_once(x)

            # print(f'output -> {output}')

            if output_seq.shape[1] == 0: #if its first prediction, retains the original value
                velocity = output - x[:, -1, :2]
                output_seq = output 
            else: #if not, it must be relative to the first prediction
                diff = output - x[:, -1, :2]
                velocity = diff
                output_seq = torch.cat((output_seq, output_seq[:, -1:, :] + diff), dim=1)
            # print(f'output_seq -> {output_seq}')

            x = torch.cat((x[:, 1:], torch.cat((output, velocity), dim=-1)), dim=1)

            # print(f'x concat -> {x}')

            x = torch.concat((x[:, :, :2] - x[:, 0:1, :2], x[:, :, 2:]), dim=-1)
        return output_seq