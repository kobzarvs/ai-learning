# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

@torch.no_grad()
def accuracy(model, x, y):
    # argmax выбирает номер категории с наибольшей оценкой.
    return float((model(x).argmax(dim=-1) == y).float().mean())

def points(count=320):
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

# Обучать сначала на простых объектах, затем постепенно добавлять сложные.
x, y = points(400)
margin = (x[:,0]+0.5*x[:,1]).abs()
order = margin[:300].argsort(descending=True)
model = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
for size in [50, 120, 300]:
    chosen = order[:size]
    fit_classifier(model, x[chosen], y[chosen], steps=50)
    print('Учебных объектов:', size, 'качество:', accuracy(model, x[300:], y[300:]))
