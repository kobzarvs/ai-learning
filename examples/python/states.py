# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Восстановить плавное скрытое значение из шумных измерений.
truth = torch.linspace(0, 2, 60)
observed = truth+0.35*torch.randn(60)
mean, variance, estimates = torch.tensor(0.), torch.tensor(1.), []
for measurement in observed:
    variance = variance+0.01  # Допускаем изменение состояния между шагами.
    gain = variance/(variance+0.35**2)
    mean = mean+gain*(measurement-mean)
    variance = (1-gain)*variance
    estimates.append(mean)
estimates = torch.stack(estimates)
print('Ошибка измерений:', F.mse_loss(observed, truth).item())
print('Ошибка фильтра:', F.mse_loss(estimates, truth).item())
