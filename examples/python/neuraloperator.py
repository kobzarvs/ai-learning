# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_DEVICES = ['cpu', 'cuda']
DEVICE_NOTE = 'Здесь используются линейная алгебра, комплексные FFT или специальные распределения; вариант MPS пока не включён.'

import argparse

parser = argparse.ArgumentParser(description="Учебный пример AI Atlas / PyTorch")
parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu",
                    help="cpu; NVIDIA cuda; Apple Metal mps; auto выбирает доступный GPU")
requested_device = parser.parse_args().device
available = {"cpu": True, "cuda": torch.cuda.is_available(),
             "mps": torch.backends.mps.is_available()}
if requested_device == "auto":
    requested_device = next(d for d in ("cuda", "mps", "cpu")
                            if d in SUPPORTED_DEVICES and available[d])
if requested_device not in SUPPORTED_DEVICES:
    parser.error("Для этого опыта доступны: " + ", ".join(SUPPORTED_DEVICES)
                 + ". " + DEVICE_NOTE)
if not available[requested_device]:
    parser.error("Устройство " + requested_device + " недоступно. "
                 "CUDA требует NVIDIA, драйвер и CUDA-сборку PyTorch; "
                 "MPS требует совместимый Mac и нативный запуск вне Linux-контейнера. "
                 "Для запуска на процессоре явно укажи --device cpu.")
device = torch.device(requested_device)
# Новые данные И параметры моделей создаются на одном выбранном устройстве.
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device, "| dtype: float32")


# Дадим модели целые записи из 32 чисел и их сглаженные версии, в которых быстрые колебания ослаблены. Модель будет учиться разделять сигнал на частоты (медленные и быстрые составляющие), менять их силу и собирать сигнал обратно. После обучения проверим, умеет ли она сглаживать новые записи. Это небольшой пример обработки сразу всей последовательности.
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
