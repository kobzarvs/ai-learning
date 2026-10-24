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

# Научить сеть отличать точки внутри круга от точек снаружи.
x = torch.rand(500, 2) * 2 - 1
y = (x.square().sum(1) < 0.45).long()
model = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 2))
fit_classifier(model, x[:400], y[:400])
print('Доля верных ответов на новых точках:', accuracy(model, x[400:], y[400:]))
