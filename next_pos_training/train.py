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

from torch.nn.utils.rnn import pad_sequence

from dataset import BallTrajectoryDataset
from network import LSTMNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 8
num_epochs = 200
learning_rate = 1e-4
weight_decay = 1e-5

init_iso_format_time = datetime.now().isoformat()
model_path = f'models/{init_iso_format_time}'
os.makedirs(model_path)
log_file = open(f"{model_path}/output.log", "w")
shutil.copy("network.py", model_path)

with open(f"{model_path}/hyperparameters.json", "w") as f:
    json.dump({
        'batch_size':batch_size,
        'num_epochs':num_epochs,
        'learning_rate':learning_rate,
        'weight_decay':weight_decay
    }, f, indent=4) 

full_dataset = BallTrajectoryDataset()
full_dataset_noiseless = BallTrajectoryDataset(noise_std=0.0)

total_size = len(full_dataset)
indices = list(range(total_size))
np.random.shuffle(indices)

split = int(np.floor(0.2 * total_size))
train_indices = indices[split:]
val_indices = indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)


def collate_fn(batch):
    past_positions, future_positions = zip(*batch)

    past_positions = [seq.clone().detach().float() for seq in past_positions]
    future_positions = [seq.clone().detach().float() for seq in future_positions]

    future_positions_padded = pad_sequence(future_positions, batch_first=True)
    original_lengths = torch.tensor([seq.shape[0] for seq in future_positions], dtype=torch.long)
    past_positions = torch.stack(past_positions)

    return past_positions, future_positions_padded, original_lengths


train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(full_dataset_noiseless, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn)

print('Loaded dataset')

model = LSTMNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

for epoch in range(num_epochs):
    model.train()  # Defina o modelo para o modo de treinamento
    for batch_idx, (past_positions, future_positions, original_lengths) in enumerate(train_loader):
        past_positions, future_positions = past_positions.to(device), future_positions.to(device)
        original_lengths = original_lengths.to(device)

        # Forward
        target_size = future_positions.shape[1]

        outputs = model(past_positions, target_size)

        mask = torch.arange(target_size, device=device).expand(len(original_lengths), target_size) < original_lengths.unsqueeze(1)

        outputs_masked = outputs[mask]

        future_positions_masked = future_positions[mask]

        loss = criterion(outputs_masked, future_positions_masked)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if (batch_idx + 1) % 10 == 0:
            epoch_log = f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]['lr']}"
            log_file.write(f'{epoch_log}\n')
            print(epoch_log)
    
    # scheduler.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (past_positions, future_positions, original_lengths) in enumerate(val_loader):
            past_positions, future_positions = past_positions.to(device), future_positions.to(device)
            original_lengths = original_lengths.to(device)

            target_size = future_positions.shape[1]
            outputs = model(past_positions, target_size)

            mask = torch.arange(target_size, device=device).expand(len(original_lengths), target_size) < original_lengths.unsqueeze(1)

            outputs_masked = outputs[mask]

            future_positions_masked = future_positions[mask]

            loss = criterion(outputs_masked, future_positions_masked)
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        validation_log = f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}"
        log_file.write(f'{validation_log}\n')
        print(validation_log)

        torch.save(model, f"{model_path}/model.pth")

print("Finished training")
