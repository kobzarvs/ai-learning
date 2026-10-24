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

# Выбрать объекты, ответы на которые полезнее всего запросить у человека.
x, y = points(320)
model = nn.Linear(2, 2)
fit_classifier(model, x[:20], y[:20])
with torch.no_grad():
    probability = model(x[20:240]).softmax(1)
    entropy = -(probability*probability.clamp_min(1e-8).log()).sum(1)
    request = entropy.topk(20).indices+20
print('Номера объектов для разметки:', request.tolist())
fit_classifier(model, torch.cat([x[:20], x[request]]), torch.cat([y[:20], y[request]]))
print('После дополнительных 20 меток:', accuracy(model, x[240:], y[240:]))
