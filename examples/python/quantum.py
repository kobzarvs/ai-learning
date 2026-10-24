# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить угол вращения одного кубита в классической симуляции.
angle = nn.Parameter(torch.tensor(0.2))
optimizer = torch.optim.Adam([angle], lr=0.05)
# После RY(theta) вероятности двух исходов: cos²(theta/2), sin²(theta/2).
for _ in range(150):
    probability_one = torch.sin(angle/2).square()
    loss = (probability_one-0.8).square()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Угол:', angle.item())
print('Вероятность исхода 1:', torch.sin(angle/2).square().item())
