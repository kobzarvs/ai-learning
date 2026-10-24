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

# Обучить сумму нелинейных функций на рёбрах описывать зависимость sin(x₁) + x₂².
x = torch.rand(400, 2) * 2 - 1
y = (x[:, :1].sin() + x[:, 1:].square())
class EdgeFunctions(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Buffer(torch.linspace(-1, 1, 12))
        self.coefficients = nn.Parameter(torch.zeros(2, 12))
    def forward(self, x):
        bases = torch.exp(-((x[..., None]-self.centers)/0.25).square())
        return (bases*self.coefficients).sum((1, 2)).unsqueeze(1)
model = EdgeFunctions()
fit_regression(model, x[:320], y[:320])
with torch.no_grad():
    print('Ошибка на новых точках:', F.mse_loss(model(x[320:]), y[320:]).item())
