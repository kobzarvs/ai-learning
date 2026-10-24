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


# Найдём кривую, которая начинается со значения 1 и убывает со скоростью, равной её текущему значению. Запись u′(x) = −u(x) означает именно это: производная u′ — скорость изменения, а минус задаёт уменьшение. Научим сеть соблюдать это правило и начальное условие u(0) = 1. Готовые значения правильной кривой используем только для проверки.
model = nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(400):
    x = torch.rand(80, 1, requires_grad=True)*2
    u = model(x)
    derivative, = torch.autograd.grad(u.sum(), x, create_graph=True)
    equation_error = (derivative+u).square().mean()
    boundary_error = (model(torch.zeros(1,1))-1).square().mean()
    loss = equation_error+boundary_error
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    x = torch.linspace(0, 2, 50)[:, None]
    print('Ошибка относительно exp(−x):', F.mse_loss(model(x), torch.exp(-x)).item())
