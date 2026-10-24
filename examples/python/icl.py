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


# В каждой задаче есть неизвестное правило y = ax + b (вход умножается на одно число, затем прибавляется другое). Дадим сети два примера «вход — правильный ответ» и третий вход, для которого нужно найти ответ. Сначала обучим её решать много таких задач. На проверке правила будут новыми, но внутренние настройки сети менять не будем: она должна использовать примеры из самого запроса.
# В каждой задаче своё правило y=a*x+b. Контекст: две пары (x,y) и новый x.
def tasks(count):
    a, b = torch.rand(count, 1)*2-1, torch.rand(count, 1)*2-1
    x1, x2 = -torch.ones(count, 1), torch.ones(count, 1)
    query = torch.rand(count, 1)*2-1
    context = torch.cat([x1, a*x1+b, x2, a*x2+b, query], 1)
    return context, a*query+b
model = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(500):
    x, y = tasks(128)
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
model.eval()
with torch.no_grad():
    x, y = tasks(100)
    print('Ошибка на новых правилах без дообучения:', F.mse_loss(model(x), y).item())
