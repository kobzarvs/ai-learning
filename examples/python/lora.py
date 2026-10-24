# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Адаптировать замороженное преобразование через две небольшие обучаемые матрицы.
x = torch.randn(300, 8)
base = torch.randn(8, 8)*0.2
true_update = torch.randn(8, 2)@torch.randn(2, 8)*0.1
y = x@(base+true_update)
a = nn.Parameter(torch.randn(8, 2)*0.01)
b = nn.Parameter(torch.zeros(2, 8))
optimizer = torch.optim.AdamW([a, b], lr=0.04)
for _ in range(250):
    prediction = x[:240]@base + x[:240]@a@b
    loss = F.mse_loss(prediction, y[:240])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    print('Ошибка на новых объектах:', F.mse_loss(x[240:]@base+x[240:]@a@b, y[240:]).item())
print('Весов основной матрицы:', base.numel(), 'обучаемых:', a.numel()+b.numel())
