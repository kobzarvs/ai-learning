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

# Измерить забывание старой задачи и влияние повторения старых примеров.
x = torch.randn(400, 2)
y_a = (x[:,0]>0).long(); y_b = (x[:,1]>0).long()
# Третий признак сообщает, какая именно задача запрошена.
a = torch.cat([x, torch.zeros(400, 1)], 1)
b = torch.cat([x, torch.ones(400, 1)], 1)
first = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, 2))
fit_classifier(first, a[:300], y_a[:300])
naive, replay = copy.deepcopy(first), copy.deepcopy(first)
fit_classifier(naive, b[:300], y_b[:300])
fit_classifier(replay, torch.cat([b[:300], a[:80]]), torch.cat([y_b[:300], y_a[:80]]))
print('Старая задача после обычного дообучения:', accuracy(naive, a[300:], y_a[300:]))
print('Старая задача с повторением:', accuracy(replay, a[300:], y_a[300:]))
