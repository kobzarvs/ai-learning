# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Проверить производную через автоматическое дифференцирование и маленькое изменение входа.
x = torch.tensor(3., requires_grad=True)
y = x.square()+2*x
y.backward()
h = 0.001
finite_difference = ((x.detach()+h).square()+2*(x.detach()+h)-y.detach())/h
print('Производная PyTorch:', x.grad.item())
print('Численная оценка:', finite_difference.item())
assert abs(x.grad.item()-8)<1e-6
