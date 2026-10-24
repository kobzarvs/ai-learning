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

# Обучить сжатие и восстановление обычных объектов, затем сравнить ошибку на необычном объекте.
normal = torch.randn(300, 1) @ torch.tensor([[1.,2.,-1.,0.5]])
normal += 0.03*torch.randn_like(normal)
model = nn.Sequential(nn.Linear(4, 1), nn.Linear(1, 4))
fit_regression(model, normal[:240], normal[:240], steps=300)
with torch.no_grad():
    ordinary = (model(normal[240:])-normal[240:]).square().mean(1)
    strange = torch.tensor([[5.,-5.,5.,-5.]])
    unusual = (model(strange)-strange).square().mean()
print('Обычные: средняя ошибка', ordinary.mean().item())
print('Необычный: ошибка', unusual.item())
