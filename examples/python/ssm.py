# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Проследить, как модель пространства состояний хранит затухающую память о входном импульсе.
x = torch.zeros(30)
x[3] = 1
state = torch.zeros(4)
decay = torch.tensor([0.1, 0.5, 0.8, 0.95])
readout = torch.tensor([0.25, 0.25, 0.25, 0.25])
outputs = []
for value in x:
    state = decay * state + value
    outputs.append((readout * state).sum().item())
print('Ответ после импульса:', [round(v, 3) for v in outputs[3:12]])
assert outputs[3] > outputs[8] > outputs[20] > 0
