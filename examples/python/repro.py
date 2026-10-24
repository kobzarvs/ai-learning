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

# Повторить обучение с одинаковой и разной инициализацией и сравнить ответы.
x = torch.randn(100, 2); y = x.sum(1, keepdim=True)
def run(seed):
    torch.manual_seed(seed)
    model = nn.Linear(2, 1)
    fit_regression(model, x, y, steps=20)
    return model(x).detach()
a, b, c = run(7), run(7), run(8)
print('Одинаковый seed:', torch.equal(a, b))
print('Разница при другом seed:', (a-c).abs().max().item())
print('Версия PyTorch:', torch.__version__)
assert torch.equal(a, b)
