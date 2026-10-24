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

# Автоматически выбрать ширину небольшой сети по отдельной проверочной выборке.
x, y = points(500)
results = []
for width in [4, 12, 32]:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, 2))
    fit_classifier(model, x[:300], y[:300], steps=80)
    score = accuracy(model, x[300:400], y[300:400])
    results.append((score, width, model))
score, width, best = max(results, key=lambda r:r[0])
print('Выбрана ширина:', width, 'по проверке:', score)
print('Окончательный тест:', accuracy(best, x[400:], y[400:]))
