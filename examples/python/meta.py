# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def fit_regression(model, x, y, steps=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

# Подготовить общую стартовую точку, из которой модель быстро подстраивается под новую прямую.
initial = nn.Linear(1, 1)
for _ in range(120):
    task = copy.deepcopy(initial)
    x = torch.randn(12, 1)
    slope = 1+torch.rand(()); bias = torch.rand(())*0.4
    fit_regression(task, x, slope*x+bias, steps=10)
    with torch.no_grad():
        for shared, adapted in zip(initial.parameters(), task.parameters()):
            shared.lerp_(adapted, 0.15)
x = torch.randn(30, 1); y = 1.7*x+0.15
adapted = copy.deepcopy(initial)
fit_regression(adapted, x[:5], y[:5], steps=10)
with torch.no_grad():
    print('До адаптации:', F.mse_loss(initial(x[5:]), y[5:]).item())
    print('После 10 шагов:', F.mse_loss(adapted(x[5:]), y[5:]).item())
