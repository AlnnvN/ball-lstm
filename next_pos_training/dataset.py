import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset

class BallTrajectoryDataset(Dataset):
    def __init__(self, input_positions_quantity:int=15, output_positions_quantity:int=1, noise_std:float=0.05, next_pos=True, using_velocity=True):
        self.data = []

        self.noise_std = noise_std

        self.using_velocity = using_velocity

        raw_dataset = np.load("../raw_dataset/ball_dataset.npy", allow_pickle=True)
        for trajectory in raw_dataset:
            sequences = self.split_into_sequences(trajectory)

            for seq in sequences:
                len_seq = len(seq)
                if len_seq <= (input_positions_quantity + output_positions_quantity):
                    continue
                
                # self.plot_sequence(seq)

                for i in range(len_seq - (input_positions_quantity + output_positions_quantity)):
                    input_positions = seq[i:i+input_positions_quantity]
                    output_positions = seq[i+input_positions_quantity:i+input_positions_quantity+1] if next_pos else seq[i+input_positions_quantity:]

                    initial_pos = input_positions[0][0]

                    input_positions = [
                        (pos[0][0] - initial_pos[0], pos[0][2] - initial_pos[2])
                        for pos in input_positions
                    ]

                    output_positions = [
                        (pos[0][0] - initial_pos[0], pos[0][2] - initial_pos[2])
                        for pos in output_positions
                    ]

                    input_positions = np.round(input_positions, 3)
                    output_positions = np.round(output_positions, 3)

                    self.data.append(
                        (np.array(input_positions, dtype=np.float32),
                        np.array(output_positions, dtype=np.float32)))

    '''
        A -> fB  | fC -> append(f)B | append(f)C 
        B -> tB | D 
        C -> fC | D
        D -> f -> append(f)
    '''

    def split_into_sequences(self, trajectory):
        sequences = []
        current_seq = []

        index = 0
        trajectory_len = len(trajectory)

        while trajectory[index][1] == False:
            
            #A -> false
            current_seq.append(trajectory[index])
            index+=1
            if index >= trajectory_len:
                break
            
            #B
            if trajectory[index][1] == True:
                while index < trajectory_len and trajectory[index][1] == True:
                    current_seq.append(trajectory[index])
                    index+=1
        
            #C
            elif trajectory[index][1] == False:
                while index < trajectory_len and trajectory[index][1] == False:
                    if index == trajectory_len-1:
                        break
                    current_seq.append(trajectory[index])
                    index+=1

            if index >= trajectory_len:
                break

            #D -> false
            current_seq.append(trajectory[index])

            if len(current_seq) > 0:
                sequences.append(current_seq)

            current_seq = []

        return sequences

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        data = torch.tensor(self.data[idx][0], dtype=torch.float32)
        label = torch.tensor(self.data[idx][1], dtype=torch.float32)
    
        # applies noise only on input position, not on flying flag
        noise = torch.randn_like(data[:, :2]) * self.noise_std
        
        noisy_data = data.clone()
        noisy_data[:, :2] += noise

        # appends velocity
        if self.using_velocity:
            velocity = torch.zeros_like(noisy_data[:, :2])
            velocity[:-1] = noisy_data[1:, :2] - noisy_data[:-1, :2]
            velocity[-1] = velocity[-2]
            noisy_data = torch.cat([noisy_data, velocity], dim=1)

        return noisy_data, label
    
    @staticmethod
    def plot_sequence(sequence):
        input_positions = np.array([(pos[0][0], pos[0][2], int(pos[1])) for pos in sequence])

        input_x = input_positions[:, 0]
        input_y = input_positions[:, 1]
        input_flying = input_positions[:, 2]

        plt.figure(figsize=(8, 6))

        plt.scatter(input_x[input_flying == 0], input_y[input_flying == 0], 
                    c='blue', s=80, label='Ground (False)')

        plt.scatter(input_x[input_flying == 1], input_y[input_flying == 1], 
                    c='red', s=80, label='Flying (True)')

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Ball Trajectory Sequence")
        plt.legend()
        plt.show()

    def plot_sample(self):

        data, target = self.__getitem__(int(random.random() * (self.__len__()-1)))
        
        input_positions = data.numpy()
        output_positions = target.numpy()

        input_x = input_positions[:, 0]
        input_y = input_positions[:, 1]

        output_x = output_positions[:, 0]
        output_y = output_positions[:, 1]

        plt.figure(figsize=(8, 6))

        plt.scatter(input_x, input_y, 
                    c='blue', s=80, label='Input Ground (False)')

        plt.scatter(output_x, output_y, c='green', s=80, label='Output')

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Ball Trajectory Sample")
        plt.legend()
        plt.show()
