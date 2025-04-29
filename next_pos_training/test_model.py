import importlib
import math
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BallTrajectoryDataset
import json

def to_float(value):
    return round(value.item() if isinstance(value, torch.Tensor) else value, 4)

plot_model = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_folder = "2025-03-30T04:20:15.997370"
model_name = "model_epoch_400.pth"

sys.path.append(f'models/{model_folder}')

spec = importlib.util.spec_from_file_location('network', f'models/{model_folder}/network.py')
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)

state_dict = torch.load(f"models/{model_folder}/models/{model_name}", map_location=device, weights_only=True)

with open(f"models/{model_folder}/parameters.json", "r", encoding="utf-8") as file:
    dados = json.load(file)

model = network.LSTMNetwork(using_velocity=dados['using_velocity'])
model.load_state_dict(state_dict)
model = model.to(device)

full_dataset = BallTrajectoryDataset(
    input_positions_quantity=dados['input_positions_quantity'], 
    next_pos=False, 
    noise_std=0.01,
    is_test=True,
    using_velocity=dados['using_velocity']
)

val_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)

criterion = nn.SmoothL1Loss()
model.eval()

if plot_model:
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
else:
    #calculate model statistics
    losses = []

    x_errors = []
    y_errors = []

    average_displacement_errors = []
    final_displacement_errors = []
    losses = []

    model.eval()
    with torch.no_grad():
        for past_positions, future_positions in val_loader:
            past_positions, future_positions = past_positions.to(device), future_positions.to(device)
            target_size = future_positions.shape[1]
            outputs = model(past_positions, target_size)

            # Compute the loss
            loss = criterion(outputs, future_positions)
            losses.append(loss.item())

            x_error = outputs[:, :, 0] - future_positions[:, :, 0]
            y_error = outputs[:, :, 1] - future_positions[:, :, 1]

            x_errors.extend(x_error.cpu().numpy().flatten())
            y_errors.extend(y_error.cpu().numpy().flatten())

            # Compute the Euclidean distance errors (per position)
            errors = torch.norm(outputs - future_positions, dim=-1)  # Compute Euclidean error (2D)

            # Compute Average Displacement Error (ADE) and Final Displacement Error (FDE) for each sequence
            average_displacement_error = errors.mean().item()  # Mean error for the entire sequence
            final_displacement_error = errors[:, -1].mean().item()  # Final position error for the sequence

            average_displacement_errors.append(average_displacement_error)
            final_displacement_errors.append(final_displacement_error)

    # Convert lists to numpy arrays
    x_errors = np.array(x_errors)
    y_errors = np.array(y_errors)
    losses = np.array(losses)
    average_displacement_errors = np.array(average_displacement_errors)
    final_displacement_errors = np.array(final_displacement_errors)

    # Calculate sequence-level statistics (ADE, FDE, and loss)
    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    median_loss = float(np.median(losses))

    mean_ade = float(np.mean(average_displacement_errors))
    std_ade = float(np.std(average_displacement_errors))
    median_ade = float(np.median(average_displacement_errors))

    mean_fde = float(np.mean(final_displacement_errors))
    std_fde = float(np.std(final_displacement_errors))
    median_fde = float(np.median(final_displacement_errors))

    # Calculate per-position statistics (mean, std, and median for each X and Y position)
    mean_x_error = float(np.mean(x_errors))
    std_x_error = float(np.std(x_errors))
    median_x_error = float(np.median(x_errors))

    mean_y_error = float(np.mean(y_errors))
    std_y_error = float(np.std(y_errors))
    median_y_error = float(np.median(y_errors))

    stats = {
        "mean_sequence_loss": to_float(mean_loss),
        "std_sequence_loss": to_float(std_loss),
        "median_sequence_loss": to_float(median_loss),

        "mean_sequence_displacement_error": to_float(mean_ade),
        "std_sequence_displacement_error": to_float(std_ade),
        "median_sequence_displacement_error": to_float(median_ade),

        "mean_final_displacement_error": to_float(mean_fde),
        "std_final_displacement_error": to_float(std_fde),
        "median_final_displacement_error": to_float(median_fde),

        "mean_x_error": to_float(mean_x_error),
        "std_x_error": to_float(std_x_error),
        "median_x_error": to_float(median_x_error),

        "mean_y_error": to_float(mean_y_error),
        "std_y_error": to_float(std_y_error),
        "median_y_error": to_float(median_y_error),

        "noise_tested_on": to_float(full_dataset.noise_std),
        "loss_function": str(criterion),
        "model_name": model_name
    }

    # Save statistics to JSON
    stats_path = f"models/{model_folder}/statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"Model statistics saved to {stats_path}")

