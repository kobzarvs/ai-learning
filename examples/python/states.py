# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_DEVICES = ['cpu', 'cuda', 'mps']
DEVICE_NOTE = 'GPU требует поддержки операций этого опыта установленной версией фреймворка. Автоматический переход на CPU при ошибке отключён.'

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


# Представь датчик, показания которого случайно отклоняются от настоящего плавно меняющегося значения. По очереди прочитаем измерения и будем сочетать предыдущий прогноз с новым показанием. Получим более плавную оценку скрытого значения (того, что хотим измерить без помех). Сравним ошибку исходных показаний и исправленной оценки.
truth = torch.linspace(0, 2, 60)
observed = truth+0.35*torch.randn(60)
mean, variance, estimates = torch.tensor(0.), torch.tensor(1.), []
for measurement in observed:
    variance = variance+0.01  # Допускаем изменение состояния между шагами.
    gain = variance/(variance+0.35**2)
    mean = mean+gain*(measurement-mean)
    variance = (1-gain)*variance
    estimates.append(mean)
estimates = torch.stack(estimates)
print('Ошибка измерений:', F.mse_loss(observed, truth).item())
print('Ошибка фильтра:', F.mse_loss(estimates, truth).item())
