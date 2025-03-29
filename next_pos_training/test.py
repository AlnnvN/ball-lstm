import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BallTrajectoryDataset
from network import LSTMNetwork
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_folder = "2025-03-29T04:23:45.364335"

model_name = "model_epoch_200.pth"

state_dict = torch.load(f"models/{model_folder}/{model_name}", map_location=device, weights_only=True)

with open(f"models/{model_folder}/parameters.json", "r", encoding="utf-8") as file:
    dados = json.load(file)

model = LSTMNetwork(using_velocity=dados['using_velocity'])
model.load_state_dict(state_dict)
model = model.to(device)

full_dataset = BallTrajectoryDataset(
    input_positions_quantity=dados['input_positions_quantity'], 
    next_pos=False, 
    noise_std=0.05,
    using_velocity=dados['using_velocity']
)

val_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)

criterion = nn.MSELoss()
model.eval()
with torch.no_grad():
    val_loss = 0
    for val_batch_idx, (past_positions, future_positions) in enumerate(val_loader):
        past_positions, future_positions = past_positions.to(device), future_positions.to(device)

        target_size = future_positions.shape[1]

        outputs = model(past_positions, target_size)

        loss = criterion(outputs, future_positions)
        val_loss = loss.item()

        future_positions = future_positions.cpu().numpy()
        outputs = outputs.cpu().numpy()

        print(f'target shape -> {future_positions.shape}')
        print(f'output shape -> {outputs.shape}')

        validation_log = f"Validation Loss: {val_loss:.4f}, SqRtLoss: {math.sqrt(val_loss):.4f}"
        print(validation_log)

        plt.figure(figsize=(6, 6))

        plt.plot(future_positions[:, :, 0].T, future_positions[:, :, 1].T, 'bo-', alpha=0.5, label="ground-truth")
        plt.plot(outputs[:, :, 0].T, outputs[:, :, 1].T, 'ro--', alpha=0.5, label="prediction")

        plt.xlabel("Posição X")
        plt.ylabel("Posição Y")
        plt.title(f"Trajetória Real vs Predita - Batch {val_batch_idx+1}")
        plt.legend()
        plt.grid(True)
        plt.show()
    


print("Finished training")
