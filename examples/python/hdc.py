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

# Закодировать объекты длинными случайными векторами и классифицировать по прототипам.
x, y = points(300)
projection = torch.randn(2, 1024)
codes = torch.where(x@projection>=0, 1., -1.)
prototypes = torch.stack([codes[:240][y[:240]==k].sum(0) for k in range(2)])
prototypes = F.normalize(prototypes, dim=1)
prediction = (F.normalize(codes[240:], dim=1)@prototypes.T).argmax(1)
print('Доля верных ответов:', prediction.eq(y[240:]).float().mean().item())
