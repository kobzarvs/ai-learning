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


# Проверим, может ли PyTorch выполнять вычисления на графическом процессоре Mac. Если может, перенесём туда маленькую модель и данные; иначе используем обычный процессор. Выполним один шаг обучения: посчитаем ошибку и изменим настройки модели. В выводе увидим, какое устройство действительно использовалось.
# device выбран общим параметром --device cpu / cuda / mps / auto выше.
model = nn.Linear(2, 1).to(device)
x = torch.randn(64, 2, device=device)
y = x.sum(1, keepdim=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(150):
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
if device.type == 'mps':
    torch.mps.synchronize()
elif device.type == 'cuda':
    torch.cuda.synchronize()
print('Устройство:', device, 'ошибка:', loss.detach().cpu().item())
