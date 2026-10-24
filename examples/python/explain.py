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

# Измерить, насколько ухудшается качество, если перемешать один признак между объектами.
x = torch.randn(400, 3); y = (2*x[:,0]+x[:,1]>0).long()
model = nn.Linear(3, 2)
fit_classifier(model, x[:300], y[:300])
baseline = accuracy(model, x[300:], y[300:])
for feature in range(3):
    broken = x[300:].clone()
    broken[:, feature] = broken[torch.randperm(100), feature]
    print('Признак', feature, 'падение качества:', baseline-accuracy(model, broken, y[300:]))
