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


# Дадим программе примеры чисел около 1.5 и попросим определить их центр. Затем начнём с других случайных чисел и будем перемещать их в сторону найденного центра, добавляя небольшие случайные толчки. Для выбора направления используем «энергию» (числовую оценку удалённости от центра, а не физическое электричество). Проверим, соберутся ли результаты около нужной области.
center = nn.Parameter(torch.tensor(0.))
optimizer = torch.optim.Adam([center], lr=0.05)
data = torch.randn(200)*0.2+1.5
for _ in range(100):
    loss = (data-center).square().mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
x = torch.randn(100)
for _ in range(80):
    x.requires_grad_(True)
    energy = 0.5*(x-center.detach()).square()
    gradient, = torch.autograd.grad(energy.sum(), x)
    x = (x-0.1*gradient+math.sqrt(0.02)*torch.randn_like(x)).detach()
print('Найденный центр:', center.item(), 'среднее образцов:', x.mean().item())
