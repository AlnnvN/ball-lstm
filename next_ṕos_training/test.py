
import math
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BallTrajectoryDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

full_dataset = BallTrajectoryDataset(noise_std=0.0)

while True:
    full_dataset.plot_sample()

# val_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)

# model = torch.load("models/2025-03-22T06:02:16.740506/model.pth").to(device)
# criterion = nn.MSELoss()
# model.eval()
# with torch.no_grad():
#     val_loss = 0
#     for val_batch_idx, (past_positions, future_positions) in enumerate(val_loader):
#         past_positions, future_positions = past_positions.to(device), future_positions.to(device)


        # outputs = model(past_positions)
        # loss = criterion(outputs, future_positions)
        # val_loss = loss.item()

        # future_positions = future_positions.cpu().numpy()
        # outputs = outputs.cpu().numpy()

        # validation_log = f"Validation Loss: {val_loss:.4f}, SqRtLoss: {math.sqrt(val_loss):.4f}"
        # print(validation_log)

        # plt.figure(figsize=(6, 6))

        # plt.plot(future_positions[:, :, 0].T, future_positions[:, :, 1].T, 'bo-', alpha=0.5, label="ground-truth")
        # plt.plot(outputs[:, :, 0].T, outputs[:, :, 1].T, 'ro--', alpha=0.5, label="prediction")

        # plt.xlabel("Posição X")
        # plt.ylabel("Posição Y")
        # plt.title(f"Trajetória Real vs Predita - Batch {val_batch_idx+1}")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    


print("Finished training")
