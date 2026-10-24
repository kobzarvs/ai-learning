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

# Использовать уверенные предсказания учителя как временные метки для неразмеченных объектов.
x, y = points(500)
model = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2))
fit_classifier(model, x[:30], y[:30])
print('Только 30 меток:', accuracy(model, x[400:], y[400:]))
with torch.no_grad():
    confidence, pseudo = model(x[30:400]).softmax(1).max(1)
    keep = confidence>0.95
augmented_x = torch.cat([x[:30], x[30:400][keep]])
augmented_y = torch.cat([y[:30], pseudo[keep]])
fit_classifier(model, augmented_x, augmented_y)
print('Добавлено временных меток:', int(keep.sum()))
print('После дообучения:', accuracy(model, x[400:], y[400:]))
