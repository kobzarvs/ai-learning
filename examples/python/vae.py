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


# Создадим облако точек рядом с координатами [1, −1]. Научим две части модели: первая переводит точку в промежуточное числовое описание с небольшой случайностью, вторая по этому описанию восстанавливает точку. Затем дадим второй части новые случайные описания и посмотрим, где окажутся созданные точки. Это маленький пример создания новых данных, а не генератор картинок.
x = torch.randn(256, 2)*0.3 + torch.tensor([1.,-1.])
encoder = nn.Linear(2, 4)
decoder = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=0.03)
for _ in range(200):
    mu, logvar = encoder(x).chunk(2, dim=1)
    logvar = logvar.clamp(-8, 8)
    z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
    reconstruction = F.mse_loss(decoder(z), x)
    kl = 0.5*(mu.square()+logvar.exp()-1-logvar).mean()
    loss = reconstruction + 0.1*kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    print('Новые точки:', decoder(torch.randn(5, 2)).tolist())
    print('Ошибка восстановления:', reconstruction.item())
