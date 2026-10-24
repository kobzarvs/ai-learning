# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def fit_regression(model, x, y, steps=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

# Прочитать последовательность чисел и предсказать их среднее.
x = torch.randn(300, 12, 1)
y = x.mean(dim=1)
class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.LSTM(1, 16, batch_first=True)
        self.output = nn.Linear(16, 1)
    def forward(self, values):
        states, _ = self.memory(values)
        return self.output(states[:, -1])
model = SequenceModel()
fit_regression(model, x[:240], y[:240])
with torch.no_grad():
    print('Ошибка на новых последовательностях:', F.mse_loss(model(x[240:]), y[240:]).item())
