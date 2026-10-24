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

# Сравнить обучение по настоящим и случайно перемешанным ответам.
x, y = points(400)
for random_labels in [False, True]:
    labels = y[:300].clone()
    if random_labels: labels = labels[torch.randperm(len(labels))]
    model = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 2))
    fit_classifier(model, x[:300], labels, steps=200)
    print('Случайные метки:', random_labels,
          'учебная часть:', accuracy(model, x[:300], labels),
          'новые объекты:', accuracy(model, x[300:], y[300:]))
