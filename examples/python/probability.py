# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обновить мнение о вероятности успеха после нескольких наблюдений.
# До опыта допускаем все вероятности равномерно: Beta(1, 1).
a, b = 1., 1.
observations = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])
a += observations.sum().item()
b += (1-observations).sum().item()
posterior = torch.distributions.Beta(a, b)
samples = posterior.sample((10000,))
interval = torch.quantile(samples, torch.tensor([0.025, 0.975]))
print('Оценка вероятности:', a/(a+b))
print('95% интервал:', interval.tolist())
