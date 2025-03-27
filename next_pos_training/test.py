import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BallTrajectoryDataset
from network import LSTMNetwork
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

full_dataset = BallTrajectoryDataset()

# while True:
#     full_dataset.plot_sample()

def collate_fn(batch):
    past_positions, future_positions = zip(*batch)

    past_positions = [seq.clone().detach().float() for seq in past_positions]
    future_positions = [seq.clone().detach().float() for seq in future_positions]

    future_positions_padded = pad_sequence(future_positions, batch_first=True)
    original_lengths = torch.tensor([seq.shape[0] for seq in future_positions], dtype=torch.long)
    past_positions = torch.stack(past_positions)

    return past_positions, future_positions_padded, original_lengths

val_loader = DataLoader(full_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


model = torch.load("models/2025-03-27T02:25:53.020733/model.pth").to(device)
model = LSTMNetwork().to(device)

criterion = nn.MSELoss()
model.eval()
with torch.no_grad():
    val_loss = 0
    for val_batch_idx, (past_positions, future_positions, original_lengths) in enumerate(val_loader):
        past_positions, future_positions = past_positions.to(device), future_positions.to(device)

        original_lengths = original_lengths.to(device)

        target_size = future_positions.shape[1]

        outputs = model(past_positions, target_size)

        mask = torch.arange(target_size, device=device).expand(len(original_lengths), target_size) < original_lengths.unsqueeze(1)

        outputs_masked = outputs[mask]

        future_positions_masked = future_positions[mask]

        loss = criterion(outputs_masked, future_positions_masked)
        val_loss = loss.item()

        future_positions_masked = future_positions_masked.cpu().numpy()
        outputs_masked = outputs_masked.cpu().numpy()

        print(f'target shape -> {future_positions_masked.shape}')
        print(f'output shape -> {outputs_masked.shape}')

        validation_log = f"Validation Loss: {val_loss:.4f}, SqRtLoss: {math.sqrt(val_loss):.4f}"
        print(validation_log)

        plt.figure(figsize=(6, 6))

        plt.plot(future_positions_masked[:, 0].T, future_positions_masked[:, 1].T, 'bo-', alpha=0.5, label="ground-truth")
        plt.plot(outputs_masked[:, 0].T, outputs_masked[:, 1].T, 'ro--', alpha=0.5, label="prediction")

        plt.xlabel("Posição X")
        plt.ylabel("Posição Y")
        plt.title(f"Trajetória Real vs Predita - Batch {val_batch_idx+1}")
        plt.legend()
        plt.grid(True)
        plt.show()
    


print("Finished training")
