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


# Создадим маленькую таблицу оценок: какие пользователи как оценили предметы. Для каждого пользователя и предмета модель подберёт короткий список чисел — вектор (числовое описание предпочтений или свойств). По двум таким описаниям она будет предсказывать оценку пары «пользователь — предмет». Проверим ответы на отложенных известных оценках.
true_users, true_items = torch.randn(12, 3), torch.randn(10, 3)
rating = true_users @ true_items.T
users, items = nn.Embedding(12, 3), nn.Embedding(10, 3)
mask = torch.rand(12, 10) < 0.8
optimizer = torch.optim.AdamW(list(users.parameters())+list(items.parameters()), lr=0.04)
for _ in range(400):
    predictions = users.weight @ items.weight.T
    loss = F.mse_loss(predictions[mask], rating[mask])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = users.weight @ items.weight.T
print('Ошибка на скрытых оценках:', F.mse_loss(prediction[~mask], rating[~mask]).item())
