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

# Обучить модель соединять две неполные подсказки от разных источников.
image_feature = torch.randn(400, 2)
text_feature = torch.randn(400, 2)
y = (image_feature[:,0]+text_feature[:,0]>0).long()
x = torch.cat([image_feature, text_feature], 1)
model = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
fit_classifier(model, x[:320], y[:320])
print('Оба источника:', accuracy(model, x[320:], y[320:]))
missing = x[320:].clone(); missing[:, 2:] = 0
print('Без второго источника:', accuracy(model, missing, y[320:]))
