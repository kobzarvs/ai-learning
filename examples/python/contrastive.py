# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Научить общий преобразователь выдавать похожие числа для двух зашумлённых видов одного объекта.
x = torch.randn(128, 8)
encoder = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 12))
optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.01)
for _ in range(100):
    a = F.normalize(encoder(x + 0.1*torch.randn_like(x)), dim=1)
    b = F.normalize(encoder(x + 0.1*torch.randn_like(x)), dim=1)
    scores = a @ b.T / 0.1
    loss = F.cross_entropy(scores, torch.arange(len(x)))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    new = torch.randn(40, 8)
    a = F.normalize(encoder(new), dim=1)
    b = F.normalize(encoder(new + 0.1*torch.randn_like(new)), dim=1)
    print('Найден свой второй вид:', (a @ b.T).argmax(1).eq(torch.arange(40)).float().mean().item())
