import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset import BallTrajectoryDataset
from network import LSTMNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
num_epochs = 100
learning_rate = 0.00003

full_dataset = BallTrajectoryDataset()

total_size = len(full_dataset)
indices = list(range(total_size))
np.random.shuffle(indices)

split = int(np.floor(0.2 * total_size))
train_indices = indices[split:]
val_indices = indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler)

model = LSTMNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()  # Defina o modelo para o modo de treinamento
    for batch_idx, (past_positions, future_positions) in enumerate(train_loader):
        past_positions, future_positions = past_positions.to(device), future_positions.to(device)
        
        # Forward
        outputs = model(past_positions)
        loss = criterion(outputs, future_positions)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, SqRtLoss: {math.sqrt(loss.item()):.4f}")

    # validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for val_batch_idx, (past_positions, future_positions) in enumerate(val_loader):
            past_positions, future_positions = past_positions.to(device), future_positions.to(device)
            outputs = model(past_positions)
            loss = criterion(outputs, future_positions)
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, SqRtLoss: {math.sqrt(avg_val_loss):.4f}")

        torch.save(model.state_dict(), "teste.pth")

print("Finished training")
