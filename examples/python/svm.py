# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def points(count=320):
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

# Обучить разделяющую прямую, которая оставляет запас до ближайших объектов.
x, labels = points(240)
y = labels.float()*2-1
model = nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
for _ in range(250):
    score = model(x[:180]).squeeze(1)
    loss = (1-y[:180]*score).clamp_min(0).mean()+0.01*model.weight.square().sum()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = (model(x[180:]).squeeze(1)>0).long()
print('Верно:', prediction.eq(labels[180:]).float().mean().item())
