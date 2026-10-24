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

# По двум числовым признакам научиться предсказывать число и проверить ответ на новых строках.
x = torch.randn(300, 2)
y = 3*x[:, :1] - 2*x[:, 1:] + 0.7 + 0.1*torch.randn(300, 1)
model = nn.Linear(2, 1)
fit_regression(model, x[:240], y[:240], steps=300)
with torch.no_grad():
    prediction = model(x[240:])
print('Ошибка на новых строках:', F.mse_loss(prediction, y[240:]).item())
print('Найденные коэффициенты:', model.weight.detach().tolist(), model.bias.item())
