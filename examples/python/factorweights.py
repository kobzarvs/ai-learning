# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Заменить большую матрицу произведением двух узких и измерить потерю точности.
weights = torch.randn(64, 4)@torch.randn(4, 32)+0.01*torch.randn(64, 32)
u, s, vh = torch.linalg.svd(weights, full_matrices=False)
rank = 4
left = u[:, :rank]*s[:rank]
right = vh[:rank]
x = torch.randn(20, 32)
print('Ошибка:', F.mse_loss((x@right.T)@left.T, x@weights.T).item())
print('Параметров до и после:', weights.numel(), left.numel()+right.numel())
