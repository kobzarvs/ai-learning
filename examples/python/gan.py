# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Понаблюдать за соревнованием генератора чисел и сети, отличающей их от настоящих примеров.
generator = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
critic = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
g_opt = torch.optim.AdamW(generator.parameters(), lr=0.01)
d_opt = torch.optim.AdamW(critic.parameters(), lr=0.01)
for _ in range(300):
    real = torch.randn(64, 1)*0.4+2
    fake = generator(torch.randn(64, 1))
    d_loss = F.binary_cross_entropy_with_logits(critic(real), torch.ones_like(real))
    d_loss += F.binary_cross_entropy_with_logits(critic(fake.detach()), torch.zeros_like(fake))
    d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
    # Генератор получает градиент через критика, но не меняет его веса.
    critic.requires_grad_(False)
    g_loss = F.binary_cross_entropy_with_logits(critic(fake), torch.ones_like(fake))
    g_opt.zero_grad(set_to_none=True); g_loss.backward(); g_opt.step()
    critic.requires_grad_(True)
with torch.no_grad():
    generated = generator(torch.randn(1000, 1))
print('Среднее и разброс:', generated.mean().item(), generated.std().item())
