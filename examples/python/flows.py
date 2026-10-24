# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить обратимое преобразование нормального шума в числа со средним 3 и разбросом 0.5.
data = torch.randn(300)*0.5+3
shift = nn.Parameter(torch.tensor(0.))
log_scale = nn.Parameter(torch.tensor(0.))
optimizer = torch.optim.Adam([shift, log_scale], lr=0.05)
for _ in range(300):
    z = (data-shift)*torch.exp(-log_scale)
    # Минус логарифм плотности с поправкой на изменение масштаба.
    loss = (0.5*z.square()+log_scale).mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Сдвиг:', shift.item(), 'масштаб:', log_scale.exp().item())
z = torch.randn(5)
samples = shift + log_scale.exp()*z
recovered = (samples-shift)/log_scale.exp()
print('Обратимость:', torch.allclose(z, recovered, atol=1e-5))
