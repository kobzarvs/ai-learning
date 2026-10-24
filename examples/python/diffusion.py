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


# Возьмём исходные числа и испортим их, прибавив случайные помехи разной силы. Модель получит испорченное число и указание силы помех, а учиться будет восстанавливать исходное значение. На новых примерах сравним ошибку до и после очистки. Это опыт с удалением шума — одной из идей диффузионных моделей, а не полный генератор изображений.
model = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(250):
    clean = torch.randn(128, 1)*0.3+2
    sigma = torch.rand(128, 1)*0.8+0.05
    noisy = clean + sigma*torch.randn_like(clean)
    loss = F.mse_loss(model(torch.cat([noisy, sigma], 1)), clean)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    clean = torch.randn(100, 1)*0.3+2
    sigma = torch.full_like(clean, 0.5)
    noisy = clean+sigma*torch.randn_like(clean)
    restored = model(torch.cat([noisy, sigma], 1))
print('До очистки:', F.mse_loss(noisy, clean).item())
print('После очистки:', F.mse_loss(restored, clean).item())
