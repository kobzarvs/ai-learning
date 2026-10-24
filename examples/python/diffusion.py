# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить модель убирать добавленный шум при нескольких уровнях зашумления.
model = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(250):
    clean = torch.randn(128, 1)*0.3+2
    sigma = torch.rand(128, 1)*0.8+0.05
    noisy = clean + sigma*torch.randn_like(clean)
    loss = F.mse_loss(model(torch.cat([noisy, sigma], 1)), clean)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    clean = torch.randn(100, 1)*0.3+2
    sigma = torch.full_like(clean, 0.5)
    noisy = clean+sigma*torch.randn_like(clean)
    restored = model(torch.cat([noisy, sigma], 1))
print('До очистки:', F.mse_loss(noisy, clean).item())
print('После очистки:', F.mse_loss(restored, clean).item())
