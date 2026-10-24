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

# Перенести обученное представление на немного изменившуюся задачу, обучая только новую голову.
x, y = points(400)
encoder = nn.Sequential(nn.Linear(2, 16), nn.SiLU())
source = nn.Sequential(encoder, nn.Linear(16, 2))
fit_classifier(source, x[:300], y[:300])
encoder.requires_grad_(False)
target = nn.Sequential(encoder, nn.Linear(16, 2))
new_labels = (x[:,0]-0.3*x[:,1]>0.2).long()
fit_classifier(target, x[:60], new_labels[:60])
print('Качество новой головы:', accuracy(target, x[300:], new_labels[300:]))
print('Обучаемых параметров:', sum(p.numel() for p in target.parameters() if p.requires_grad))
