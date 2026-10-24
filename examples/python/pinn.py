# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить сеть находить функцию по уравнению u′(x) = −u(x) и условию u(0) = 1.
model = nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(400):
    x = torch.rand(80, 1, requires_grad=True)*2
    u = model(x)
    derivative, = torch.autograd.grad(u.sum(), x, create_graph=True)
    equation_error = (derivative+u).square().mean()
    boundary_error = (model(torch.zeros(1,1))-1).square().mean()
    loss = equation_error+boundary_error
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    x = torch.linspace(0, 2, 50)[:, None]
    print('Ошибка относительно exp(−x):', F.mse_loss(model(x), torch.exp(-x)).item())
