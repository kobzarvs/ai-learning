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


# Возьмём числа, из которых состоят веса модели (её внутренние настройки). Попробуем хранить каждое в одном байте, используя только 256 целочисленных значений вместо дробных чисел. Затем приблизительно восстановим исходные веса и измерим потерю точности и размер хранения. Это проверка упаковки, а не доказательство ускорения всей модели.
weights = torch.randn(128, 64)*0.1
scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)/127
packed = (weights/scale).round().clamp(-127, 127).to(torch.int8)
restored = packed.float()*scale
x = torch.randn(20, 64)
print('Ошибка выхода:', F.mse_loss(x@restored.T, x@weights.T).item())
print('Байт до:', weights.numel()*weights.element_size())
print('Байт после с масштабами:', packed.numel()*packed.element_size()+scale.numel()*scale.element_size())
