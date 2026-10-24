# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Научить поле скоростей переносить облако шума к облаку данных.
velocity = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 2))
optimizer = torch.optim.AdamW(velocity.parameters(), lr=0.01)
for _ in range(350):
    source = torch.randn(128, 2)
    target = torch.randn(128, 2)*0.3 + torch.tensor([2.,-1.])
    time = torch.rand(128, 1)
    position = (1-time)*source + time*target
    loss = F.mse_loss(velocity(torch.cat([position, time], 1)), target-source)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    samples = torch.randn(500, 2)
    for step in range(60):
        time = torch.full((500, 1), step/60)
        samples += velocity(torch.cat([samples, time], 1))/60
print('Среднее новых точек:', samples.mean(0).tolist())
print('Разброс:', samples.std(0).tolist())
