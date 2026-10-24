# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить только выходной слой, оставив случайную рекуррентную память неизменной.
signal = torch.sin(torch.arange(260)*0.13)
w = torch.randn(24, 24)*0.08
input_weights = torch.randn(24)*0.3
state, states = torch.zeros(24), []
for value in signal[:-1]:
    state = (w @ state + input_weights*value).tanh()
    states.append(state)
x, y = torch.stack(states), signal[1:, None]
# Считаем веса выхода по учебной части; малый штраф стабилизирует решение.
a, b = x[20:200], y[20:200]
readout = torch.linalg.solve(a.T@a + 0.001*torch.eye(24), a.T@b)
print('Ошибка на более позднем участке:', F.mse_loss(x[200:]@readout, y[200:]).item())
