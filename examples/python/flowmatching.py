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


# Создадим два облака точек: случайные точки около нуля и учебные точки около [2, −1]. Научим сеть подсказывать, куда и насколько сдвигать точку на каждом этапе пути от первого облака ко второму. Эти подсказки называют полем скоростей (направление и скорость движения в каждой точке). Затем начнём с нового случайного облака и посмотрим, куда сеть его переместит.
velocity = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 2))
optimizer = torch.optim.AdamW(velocity.parameters(), lr=0.01)
for _ in range(350):
    source = torch.randn(128, 2)
    target = torch.randn(128, 2)*0.3 + torch.tensor([2.,-1.])
    time = torch.rand(128, 1)
    position = (1-time)*source + time*target
    loss = F.mse_loss(velocity(torch.cat([position, time], 1)), target-source)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    samples = torch.randn(500, 2)
    for step in range(60):
        time = torch.full((500, 1), step/60)
        samples += velocity(torch.cat([samples, time], 1))/60
print('Среднее новых точек:', samples.mean(0).tolist())
print('Разброс:', samples.std(0).tolist())
