import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BallTrajectoryDataset


full_dataset = BallTrajectoryDataset(
    input_positions_quantity=25, 
    next_pos=False, 
    noise_std=0.025,
    using_velocity=False
)

while True:
    full_dataset.plot_sample()