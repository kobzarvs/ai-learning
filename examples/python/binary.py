# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Заменить веса на два значения и увидеть, какую точность теряет крайнее сжатие.
weights = torch.randn(32, 16)
scale = weights.abs().mean(1, keepdim=True)
binary = torch.where(weights>=0, 1., -1.)*scale
x = torch.randn(40, 16)
print('Ошибка выхода:', F.mse_loss(x@binary.T, x@weights.T).item())
print('Теоретические биты знаков:', weights.numel())
print('Фактические байты этого float32-тензора:', binary.numel()*binary.element_size())
