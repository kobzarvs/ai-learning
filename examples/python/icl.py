# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить сеть читать примеры задачи из входа и отвечать без обновления весов на проверке.
# В каждой задаче своё правило y=a*x+b. Контекст: две пары (x,y) и новый x.
def tasks(count):
    a, b = torch.rand(count, 1)*2-1, torch.rand(count, 1)*2-1
    x1, x2 = -torch.ones(count, 1), torch.ones(count, 1)
    query = torch.rand(count, 1)*2-1
    context = torch.cat([x1, a*x1+b, x2, a*x2+b, query], 1)
    return context, a*query+b
model = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(500):
    x, y = tasks(128)
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
model.eval()
with torch.no_grad():
    x, y = tasks(100)
    print('Ошибка на новых правилах без дообучения:', F.mse_loss(model(x), y).item())
