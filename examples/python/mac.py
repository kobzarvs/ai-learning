# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Выбрать доступное устройство Mac и выполнить на нём один настоящий шаг обучения.
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = nn.Linear(2, 1).to(device)
x = torch.randn(64, 2, device=device)
y = x.sum(1, keepdim=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(150):
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
if device.type == 'mps':
    torch.mps.synchronize()
print('Устройство:', device, 'ошибка:', loss.detach().cpu().item())
