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

# Обучить модель следующего состояния и с её помощью выбрать действие к цели.
state = torch.randn(400, 1)
action = torch.rand(400, 1)*2-1
next_state = state+0.3*action
model = nn.Linear(2, 1)
fit_regression(model, torch.cat([state[:320], action[:320]], 1), next_state[:320])
with torch.no_grad():
    candidates = torch.linspace(-1, 1, 21)[:, None]
    current = torch.full_like(candidates, 0.8)
    forecast = model(torch.cat([current, candidates], 1))
    chosen = candidates[forecast.square().argmin()]
    error = F.mse_loss(model(torch.cat([state[320:], action[320:]], 1)), next_state[320:])
print('Ошибка модели мира:', error.item(), 'действие к нулю:', chosen.item())
