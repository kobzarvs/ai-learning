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


# Возьмём точки с двумя координатами, расположенные почти вдоль одной прямой. Попробуем заменить две координаты одним числом — положением вдоль главного направления облака. Затем восстановим приблизительные точки и измерим, сколько информации потеряли. Так выглядит простой пример уменьшения числа признаков (чисел, которыми описывается объект).
t = torch.linspace(-2, 2, 120)
x = torch.stack([t, 2*t, -t], dim=1)+0.05*torch.randn(120, 3)
mean = x[:90].mean(0)
u, s, vh = torch.linalg.svd(x[:90]-mean, full_matrices=False)
direction = vh[:1]
compressed = (x[90:]-mean) @ direction.T
restored = compressed @ direction + mean
print('Размер до и после:', tuple(x[90:].shape), tuple(compressed.shape))
print('Ошибка восстановления:', F.mse_loss(restored, x[90:]).item())
