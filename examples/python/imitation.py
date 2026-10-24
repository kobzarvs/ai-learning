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

# Научить сеть повторять простое управление движением к нулю.
x = torch.randn(400, 2)  # положение и скорость
expert = (-0.7*x[:, :1]-0.3*x[:, 1:]).clamp(-1, 1)
model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Tanh())
fit_regression(model, x[:320], expert[:320])
with torch.no_grad():
    print('Ошибка повторения эксперта:', F.mse_loss(model(x[320:]), expert[320:]).item())
    position, speed = 2., 0.
    for _ in range(60):
        action = float(model(torch.tensor([[position, speed]])))
        speed += 0.1*action
        position += 0.1*speed
print('Положение после управления:', position)
