# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Упаковать веса в 8-битные целые числа и измерить ошибку обратного восстановления.
weights = torch.randn(128, 64)*0.1
scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)/127
packed = (weights/scale).round().clamp(-127, 127).to(torch.int8)
restored = packed.float()*scale
x = torch.randn(20, 64)
print('Ошибка выхода:', F.mse_loss(x@restored.T, x@weights.T).item())
print('Байт до:', weights.numel()*weights.element_size())
print('Байт после с масштабами:', packed.numel()*packed.element_size()+scale.numel()*scale.element_size())
