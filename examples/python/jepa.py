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


def fit_regression(model, x, y, steps=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

# Создадим связанные части одного объекта: по двум видимым числам можно вычислить два скрытых. Скрытую часть заранее переведём в описание из восьми чисел и оставим это правило неизменным. Научим сеть по видимой части угадывать именно такое описание. Здесь «представление» означает числовое описание объекта; настоящих изображений или видео в опыте нет.
x = torch.randn(256, 4)
context = x[:, :2]
target_part = torch.stack([x[:,0]+x[:,1], x[:,0]-x[:,1]], 1)
# Фиксированный целевой encoder не может схлопнуться в одну константу.
target_encoder = nn.Linear(2, 8, bias=False).requires_grad_(False)
predictor = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 8))
with torch.no_grad():
    target = target_encoder(target_part)
fit_regression(predictor, context[:200], target[:200])
with torch.no_grad():
    print('Ошибка предсказания представления:', F.mse_loss(predictor(context[200:]), target[200:]).item())
