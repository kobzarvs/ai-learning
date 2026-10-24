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

# Классифицировать новую точку по трём ближайшим размеченным соседям.
x, y = points(240)
distance = torch.cdist(x[180:], x[:180])
neighbors = distance.topk(3, largest=False).indices
prediction = y[:180][neighbors].float().mean(1).ge(0.5).long()
print('Верно на новых точках:', prediction.eq(y[180:]).float().mean().item())
print('Соседи первой новой точки:', neighbors[0].tolist())
