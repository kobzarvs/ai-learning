# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Сжать почти двумерные данные до одной главной координаты и измерить потерю.
t = torch.linspace(-2, 2, 120)
x = torch.stack([t, 2*t, -t], dim=1)+0.05*torch.randn(120, 3)
mean = x[:90].mean(0)
u, s, vh = torch.linalg.svd(x[:90]-mean, full_matrices=False)
direction = vh[:1]
compressed = (x[90:]-mean) @ direction.T
restored = compressed @ direction + mean
print('Размер до и после:', tuple(x[90:].shape), tuple(compressed.shape))
print('Ошибка восстановления:', F.mse_loss(restored, x[90:]).item())
