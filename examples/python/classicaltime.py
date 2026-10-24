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

# Предсказать следующий отсчёт по предыдущим шести, проверяя только более поздние наблюдения.
signal = torch.sin(torch.arange(300)*0.1)+0.05*torch.randn(300)
windows = signal.unfold(0, 7, 1)
x, y = windows[:, :6], windows[:, 6:]
model = nn.Linear(6, 1)
fit_regression(model, x[:220], y[:220])
with torch.no_grad():
    prediction = model(x[220:])
print('Ошибка модели:', F.mse_loss(prediction, y[220:]).item())
print('Ошибка «повтори последнее»:', F.mse_loss(x[220:, -1:], y[220:]).item())
