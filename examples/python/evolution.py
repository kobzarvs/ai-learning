# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Найти два параметра без производных: сохранять лучшие решения и слегка изменять их.
population = torch.randn(100, 2)*3
target = torch.tensor([1.5, -0.7])
for _ in range(50):
    score = (population-target).square().sum(1)
    elite = population[score.topk(10, largest=False).indices]
    children = elite.repeat(9, 1)+0.08*torch.randn(90, 2)
    population = torch.cat([elite, children])
best = population[((population-target).square().sum(1)).argmin()]
print('Найденные параметры:', best.tolist())
