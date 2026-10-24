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

# Обучить предсказатель восстанавливать скрытое представление другой части объекта.
x = torch.randn(256, 4)
context = x[:, :2]
target_part = torch.stack([x[:,0]+x[:,1], x[:,0]-x[:,1]], 1)
# Фиксированный целевой encoder не может схлопнуться в одну константу.
target_encoder = nn.Linear(2, 8, bias=False).requires_grad_(False)
predictor = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 8))
with torch.no_grad():
    target = target_encoder(target_part)
fit_regression(predictor, context[:200], target[:200])
with torch.no_grad():
    print('Ошибка предсказания представления:', F.mse_loss(predictor(context[200:]), target[200:]).item())
