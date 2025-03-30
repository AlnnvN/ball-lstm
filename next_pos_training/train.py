import json
import math
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt

from dataset import BallTrajectoryDataset
from network import LSTMNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 16
num_epochs = 400
learning_rate = 1e-3
weight_decay = 1e-5
input_positions_quantity = 15
noise_std = 0.05
using_velocity = True

init_iso_format_time = datetime.now().isoformat()
model_path = f'models/{init_iso_format_time}'
os.makedirs(model_path)
os.makedirs(f'{model_path}/models')
log_file = open(f"{model_path}/output.log", "w")
shutil.copy("network.py", model_path)

with open(f"{model_path}/parameters.json", "w") as f:
    json.dump({
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'input_positions_quantity': input_positions_quantity,
        'noise_std': noise_std,
        'using_velocity': using_velocity
    }, f, indent=4)

full_dataset = BallTrajectoryDataset(input_positions_quantity=input_positions_quantity, noise_std=noise_std, using_velocity=using_velocity)
full_dataset_noiseless = BallTrajectoryDataset(input_positions_quantity=input_positions_quantity, noise_std=0.0, using_velocity=using_velocity)

total_size = len(full_dataset)
indices = list(range(total_size))
np.random.shuffle(indices)

split = int(np.floor(0.2 * total_size))
train_indices = indices[split:]
val_indices = indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(full_dataset_noiseless, batch_size=batch_size, sampler=val_sampler)

print('Loaded dataset')
print(f"Train size -> {len(train_loader.sampler)}")
print(f"Validation size -> {len(val_loader.sampler)}")

model = LSTMNetwork(using_velocity=using_velocity).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

train_losses = []
val_losses = []

try:
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        for batch_idx, (past_positions, future_positions) in enumerate(train_loader):
            past_positions, future_positions = past_positions.to(device), future_positions.to(device)

            outputs = model.forward_once(past_positions)
            loss = criterion(outputs, future_positions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                epoch_log = f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.7f}, Learning Rate: {optimizer.param_groups[0]['lr']}"
                log_file.write(f'{epoch_log}\n')
                print(epoch_log)
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (past_positions, future_positions) in enumerate(val_loader):
                past_positions, future_positions = past_positions.to(device), future_positions.to(device)
                outputs = model.forward_once(past_positions)
                loss = criterion(outputs, future_positions)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        validation_log = f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.7f}, Validation Loss: {avg_val_loss:.7f}"
        log_file.write(f'{validation_log}\n')
        print(validation_log)
        
        # scheduler.step() 
        
        torch.save(model.state_dict(), f"{model_path}/models/model_epoch_{epoch+1}.pth")
except KeyboardInterrupt:
    print("Finishing")


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(f"{model_path}/loss_curve.png")
plt.show()

log_file.close()
print("Finished training")
