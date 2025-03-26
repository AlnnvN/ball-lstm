import torch
import numpy as np
from torch.utils.data import Dataset

class BallTrajectoryDataset(Dataset):
    def __init__(self, input_positions_quantity:int=25, output_positions_quantity:int=100, noise_std:float=0.05):
        self.data = []

        self.noise_std = noise_std

        raw_dataset = np.load("../raw_dataset/ball_dataset.npy", allow_pickle=True)
        for trajectory in raw_dataset:
            len_trajectory = len(trajectory)
            if(len_trajectory <= input_positions_quantity):
                print('Invalid trajectory used in dataset.')

            for i in range(len_trajectory - input_positions_quantity): #acts as a slider through the positions
                input_positions = trajectory[i:i+input_positions_quantity] #selected range
                output_positions = trajectory[i+input_positions_quantity:]

                #makes sure output has output_positions_quantity positions. if lesser, just repeats the final position
                if len(output_positions) > output_positions_quantity: 
                    output_positions = output_positions[:output_positions_quantity]
                elif len(output_positions) < output_positions_quantity:
                    output_positions += [output_positions[-1]] * (output_positions_quantity - len(output_positions))

                input_positions = [
                    (pos[0][0] - input_positions[0][0][0], pos[0][2], int(pos[1]))
                    for pos in input_positions
                ]

                output_positions = [
                    (pos[0][0] - output_positions[0][0][0], pos[0][2]) 
                    for pos in output_positions
                ]

                input_positions = np.round(input_positions, 3)
                output_positions = np.round(output_positions, 3)

                self.data.append(
                    (np.array(input_positions, dtype=np.float32), 
                     np.array(output_positions, dtype=np.float32)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        data = torch.tensor(self.data[idx][0])
        label = torch.tensor(self.data[idx][1])
    
        # applies noise only on input position, not on flying flag
        noise = torch.randn_like(data[:, :2]) * self.noise_std
        
        noisy_data = data.clone()
        noisy_data[:, :2] += noise

        return noisy_data, label