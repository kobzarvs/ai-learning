# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Представить данные как смесь двух распределений и оценить, к какой части относится каждая точка.
x = torch.cat([torch.randn(120)*0.3-1.5, torch.randn(120)*0.3+1.5])
means = torch.tensor([-1., 1.]); variance = torch.ones(2); weights = torch.ones(2)/2
for _ in range(30):
    score = -0.5*((x[:, None]-means).square()/variance+variance.log())+weights.log()
    responsibility = score.softmax(1)
    count = responsibility.sum(0)
    means = (responsibility*x[:, None]).sum(0)/count
    variance = (responsibility*(x[:, None]-means).square()).sum(0)/count
    variance = variance.clamp_min(1e-4)
    weights = count/len(x)
print('Центры:', means.tolist(), 'доли:', weights.tolist())
