# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

@torch.no_grad()
def accuracy(model, x, y):
    # argmax выбирает номер категории с наибольшей оценкой.
    return float((model(x).argmax(dim=-1) == y).float().mean())

# По короткому сигналу отличить медленные колебания от быстрых.
t = torch.linspace(0, 1, 64)
y = torch.randint(0, 2, (300,))
x = torch.sin(2 * math.pi * (3 + 6*y[:, None]) * t)
x = (x + 0.15 * torch.randn_like(x)).unsqueeze(1)
model = nn.Sequential(nn.Conv1d(1, 12, 5, dilation=2), nn.SiLU(),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, 2))
fit_classifier(model, x[:240], y[:240])
print('Доля верных ответов:', accuracy(model, x[240:], y[240:]))
