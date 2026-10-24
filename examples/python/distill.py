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

# Обучить маленького ученика повторять вероятности более широкой модели.
x, y = points(400)
teacher = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
fit_classifier(teacher, x[:300], y[:300])
teacher.eval().requires_grad_(False)
student = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.03)
temperature = 2.
for _ in range(180):
    with torch.no_grad():
        soft_target = (teacher(x[:300])/temperature).softmax(1)
    soft_loss = F.kl_div((student(x[:300])/temperature).log_softmax(1), soft_target, reduction='batchmean')*temperature**2
    loss = 0.7*soft_loss+0.3*F.cross_entropy(student(x[:300]), y[:300])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Учитель:', accuracy(teacher, x[300:], y[300:]))
print('Ученик:', accuracy(student, x[300:], y[300:]))
