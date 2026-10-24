# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить вероятностный автоэнкодер на точках и получить новые точки из скрытого пространства.
x = torch.randn(256, 2)*0.3 + torch.tensor([1.,-1.])
encoder = nn.Linear(2, 4)
decoder = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=0.03)
for _ in range(200):
    mu, logvar = encoder(x).chunk(2, dim=1)
    logvar = logvar.clamp(-8, 8)
    z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
    reconstruction = F.mse_loss(decoder(z), x)
    kl = 0.5*(mu.square()+logvar.exp()-1-logvar).mean()
    loss = reconstruction + 0.1*kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    print('Новые точки:', decoder(torch.randn(5, 2)).tolist())
    print('Ошибка восстановления:', reconstruction.item())
