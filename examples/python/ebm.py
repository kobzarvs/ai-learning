# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Найти вероятные состояния по обучаемой энергии, которая низкая около наблюдаемых данных.
center = nn.Parameter(torch.tensor(0.))
optimizer = torch.optim.Adam([center], lr=0.05)
data = torch.randn(200)*0.2+1.5
for _ in range(100):
    loss = (data-center).square().mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
x = torch.randn(100)
for _ in range(80):
    x.requires_grad_(True)
    energy = 0.5*(x-center.detach()).square()
    gradient, = torch.autograd.grad(energy.sum(), x)
    x = (x-0.1*gradient+math.sqrt(0.02)*torch.randn_like(x)).detach()
print('Найденный центр:', center.item(), 'среднее образцов:', x.mean().item())
