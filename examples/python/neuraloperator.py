# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить преобразование целого сигнала в сглаженный сигнал через частоты.
x = torch.randn(180, 32)
freq = torch.fft.rfft(x)
response = torch.exp(-torch.arange(freq.shape[1]).float()/4)
y = torch.fft.irfft(freq*response, n=32)
weights = nn.Parameter(torch.ones(freq.shape[1], dtype=torch.complex64))
optimizer = torch.optim.Adam([weights], lr=0.05)
for _ in range(180):
    prediction = torch.fft.irfft(torch.fft.rfft(x[:140])*weights, n=32)
    loss = F.mse_loss(prediction, y[:140])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = torch.fft.irfft(torch.fft.rfft(x[140:])*weights, n=32)
print('Ошибка на новых сигналах:', F.mse_loss(prediction, y[140:]).item())
