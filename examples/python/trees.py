# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def points(count=320):
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

# Подобрать первое правило дерева: по какому признаку и порогу разделить объекты.
x, y = points(240)
best = None
for feature in range(2):
    for threshold in torch.linspace(-2, 2, 30):
        left = x[:180, feature] < threshold
        if not left.any() or left.all():
            continue
        a = y[:180][left].float().mean() >= 0.5
        b = y[:180][~left].float().mean() >= 0.5
        predicted = torch.where(left, a, b)
        error = (predicted != y[:180]).float().mean().item()
        if best is None or error < best[0]:
            best = (error, feature, threshold, a, b)
_, feature, threshold, a, b = best
predicted = torch.where(x[180:, feature] < threshold, a, b)
print('Признак:', feature, 'порог:', threshold.item())
print('Верно на новых строках:', (predicted == y[180:]).float().mean().item())
