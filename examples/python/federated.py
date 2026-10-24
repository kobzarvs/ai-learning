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

# Обучить локальные копии модели на двух наборах и усреднить их параметры.
x, y = points(400)
global_model = nn.Linear(2, 2)
clients = [(x[:150], y[:150]), (x[150:300], y[150:300])]
for _ in range(15):
    local_models = []
    for local_x, local_y in clients:
        local = copy.deepcopy(global_model)
        fit_classifier(local, local_x, local_y, steps=5)
        local_models.append(local)
    with torch.no_grad():
        for name, parameter in global_model.named_parameters():
            parameter.copy_(torch.stack([dict(m.named_parameters())[name] for m in local_models]).mean(0))
print('Качество общей модели:', accuracy(global_model, x[300:], y[300:]))
