# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Разделить точки на группы без заранее заданных названий категорий.
x = torch.cat([torch.randn(100, 2)*0.3-1, torch.randn(100, 2)*0.3+1])
centers = x[torch.randperm(len(x))[:2]].clone()
for _ in range(20):
    labels = torch.cdist(x, centers).argmin(1)
    for k in range(2):
        if (labels == k).any():
            centers[k] = x[labels == k].mean(0)
print('Центры групп:', centers.tolist())
print('Число точек в группах:', torch.bincount(labels, minlength=2).tolist())
