# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Увидеть, как непрерывное состояние меняется при разных интервалах между наблюдениями.
class ContinuousCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Linear(2, 4)
        self.speed = nn.Linear(2, 4)
    def forward(self, state, x, dt):
        target = self.target(x).tanh()
        decay = torch.exp(-F.softplus(self.speed(x))*dt)
        return decay*state + (1-decay)*target
cell = ContinuousCell()
x, state = torch.ones(1, 2), torch.zeros(1, 4)
for dt in [0.01, 0.1, 1.0]:
    print('Интервал', dt, 'состояние', cell(state, x, dt).detach().tolist())
