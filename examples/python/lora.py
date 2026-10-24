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


# Представь готовую часть модели, которая по восьми входным числам вычисляет восемь выходных. Её настройки хранятся в матрице 8 × 8 (таблице из 64 чисел). Оставим их замороженными — то есть запретим менять при обучении. Чтобы подстроить ответы под новую задачу, добавим две маленькие обучаемые таблицы 8 × 2 и 2 × 8: они будут рассчитывать поправку к исходному ответу. На примерах обучим только эти 32 числа и проверим, помогает ли поправка на новых входах. Это учебная модель принципа LoRA; готовую языковую модель здесь не скачиваем.
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
