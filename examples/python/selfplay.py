# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучать стратегию против собственного среднего прошлого поведения в игре «камень, ножницы, бумага».
payoff = torch.tensor([[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]])
counts = torch.tensor([9., 3., 1.])
for _ in range(2000):
    opponent = counts/counts.sum()
    # Лучший ответ на среднюю стратегию собственных прошлых игр.
    action = int((payoff@opponent).argmax())
    counts[action] += 1
average = counts/counts.sum()
print('Средняя стратегия:', average.tolist())
print('Выгода лучшего ответа соперника:', (payoff@average).max().item())
assert (payoff@average).max().item() < 0.05
