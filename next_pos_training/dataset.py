import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset

class BallTrajectoryDataset(Dataset):
    def __init__(self, input_positions_quantity:int=15, output_positions_quantity:int=15, noise_std:float=0.025, is_test=False, using_velocity=False, full_trajectory=False):
        self.input_output_positions = []

        self.noise_std = noise_std

        self.using_velocity = using_velocity

        raw_dataset = np.load("../raw_dataset/ball_dataset_train_validation.npy", allow_pickle=True) if not is_test else np.load("../raw_dataset/ball_dataset_test.npy", allow_pickle=True)
        for trajectory in raw_dataset:
            sequences = self.split_into_sequences(trajectory)

            sequences = [[pos for pos, flag in sequence] for sequence in sequences]

            for seq in sequences:
                
                #converting dt from 0.02 to 0.04
                seq = seq[::2]

                sliding_window_size = (input_positions_quantity + output_positions_quantity)

                total_sequence_size = len(seq)

                if total_sequence_size <= sliding_window_size: #if sequence does not support sliding window, it is not used.
                    continue
                
                # self.plot_sequence(seq)

                margin_to_slide = total_sequence_size - sliding_window_size

                for input_starting_position in range(margin_to_slide):
                    output_starting_position = input_starting_position+input_positions_quantity

                    input_positions = seq[input_starting_position:output_starting_position]

                    if full_trajectory:
                        output_positions = seq[output_starting_position:]
                    else:
                        output_positions = seq[output_starting_position:output_starting_position+output_positions_quantity]

                    initial_pos = input_positions[0]

                    input_positions = [
                        (pos[0] - initial_pos[0], pos[2] - initial_pos[2])
                        for pos in input_positions
                    ]

                    output_positions = [
                        (pos[0] - initial_pos[0], pos[2] - initial_pos[2])
                        for pos in output_positions
                    ]

                    input_positions = np.round(input_positions, 3)
                    output_positions = np.round(output_positions, 3)

                    self.input_output_positions.append(
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
        return len(self.input_output_positions)
    
    def __getitem__(self, idx):

        input = torch.tensor(self.input_output_positions[idx][0], dtype=torch.float32)
        raw_output = torch.tensor(self.input_output_positions[idx][1], dtype=torch.float32)
    
        # applies noise only on input position
        noise = torch.randn_like(input) * self.noise_std
        
        noisy_input = input.clone()
        noisy_input += noise

        # appends velocity
        if self.using_velocity:
            velocity = torch.zeros_like(noisy_input)
            velocity[:-1] = noisy_input[1:] - noisy_input[:-1]
            velocity[-1] = velocity[-2]
            noisy_input = torch.cat([noisy_input, velocity], dim=1)

        return noisy_input, raw_output
    
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

if __name__ == "__main__":
    dataset = BallTrajectoryDataset()
    dataset.plot_sample()