import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definição da rede LSTM para prever vários passos à frente
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Agora gera múltiplos outputs

    def forward(self, x):
        out, _ = self.lstm(x,)
        out = self.fc(out[:, -1, :])  # Pega a última saída da sequência
        return out.abs()

# Função para gerar dados de cosseno absoluto
def generate_abs_cosine_data(seq_length, pred_length, num_samples, noise_level=0):
    x = np.linspace(0, 50, num_samples)
    y = np.abs(np.cos(x))  # |cos(x)|

    # Adiciona um pequeno ruído
    y += noise_level * np.random.randn(len(y))

    sequences, targets = [], []
    for i in range(len(y) - seq_length - pred_length):
        sequences.append(y[i:i+seq_length])  # Entrada: janela deslizante
        targets.append(y[i+seq_length:i+seq_length+pred_length])  # Saída: múltiplos valores

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# Parâmetros do dataset
seq_length = 10   # Tamanho da janela de entrada
pred_length = 100 
num_samples = 300

# Criando dataset
train_x, train_y = generate_abs_cosine_data(seq_length, pred_length, num_samples)
train_size = int(len(train_x) * 0.8)

test_x, test_y = train_x[train_size:], train_y[train_size:]
train_x, train_y = train_x[:train_size], train_y[:train_size]

# Convertendo para PyTorch e adicionando dimensão extra para entrada LSTM
train_x, train_y = torch.tensor(train_x).unsqueeze(-1), torch.tensor(train_y)
test_x, test_y = torch.tensor(test_x).unsqueeze(-1), torch.tensor(test_y)

# Hiperparâmetros ajustados
input_size = 1
hidden_size = 30
num_layers = 1
output_size = pred_length  
learning_rate = 0.005
num_epochs = 200

# Criando modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Movendo dados para GPU se disponível
train_x, train_y = train_x.to(device), train_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)

# Treinamento
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Avaliação
model.eval()
with torch.no_grad():
    test_preds = model(test_x).detach().cpu().numpy()
    actual = test_y.cpu().numpy()

# Plotando previsões
plt.figure(figsize=(10, 5))
plt.plot(actual[25], label="Real", marker="o")
plt.plot(test_preds[25], label="Previsto", marker="s")
plt.legend()
plt.title("Previsão de múltiplos passos à frente para |cos(x)|")
plt.show()
