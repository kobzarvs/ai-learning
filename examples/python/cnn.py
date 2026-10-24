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

def stripes(count=240):
    # 0 = вертикальная полоса, 1 = горизонтальная. Остальное — шум.
    y = torch.randint(0, 2, (count,))
    x = torch.randn(count, 1, 8, 8) * 0.12
    for i, label in enumerate(y):
        pos = int(torch.randint(1, 6, (1,)))
        if label == 0:
            x[i, 0, :, pos:pos+2] += 1
        else:
            x[i, 0, pos:pos+2, :] += 1
    return x, y

# Обучить небольшую сеть различать вертикальные и горизонтальные полосы на картинках 8 × 8.
x, y = stripes()
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2),
)
fit_classifier(model, x[:180], y[:180])
print('Верно на новых картинках:', accuracy(model, x[180:], y[180:]))
print('Обучаемых чисел:', sum(p.numel() for p in model.parameters()))
