# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Проследить, как размер шага влияет на поиск минимума ошибки.
for learning_rate in [0.05, 0.4, 1.1]:
    parameter = nn.Parameter(torch.tensor(5.))
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    losses = []
    for _ in range(20):
        loss = (parameter-2).square()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    print('Шаг:', learning_rate, 'первые ошибки:', losses[:3], 'последняя:', losses[-1])
