# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить сеть восстанавливать скрытые элементы, не создавая ручные метки.
t = torch.randn(320, 1)
x = torch.cat([t, 2*t, -t, 0.5*t], 1)+0.03*torch.randn(320, 4)
model = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 4))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(220):
    mask = torch.rand(250, 4)<0.35
    observed = x[:250].masked_fill(mask, 0)
    prediction = model(torch.cat([observed, mask.float()], 1))
    loss = ((prediction-x[:250]).square()*mask).sum()/mask.sum().clamp_min(1)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    mask = torch.zeros(70, 4, dtype=torch.bool); mask[:, 1] = True
    prediction = model(torch.cat([x[250:].masked_fill(mask, 0), mask.float()], 1))
print('Ошибка на скрытом признаке новых объектов:', F.mse_loss(prediction[:,1], x[250:,1]).item())
