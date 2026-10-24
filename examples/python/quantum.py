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


# На обычном компьютере рассчитаем поведение одного кубита (квантового элемента, измерение которого даёт 0 или 1 с определёнными вероятностями). Будем менять один угол в формуле, пока вероятность получить 1 не станет близкой к 80%. Это обучение одного числового параметра в симуляции. Настоящего квантового устройства и доказательства его преимуществ в этом опыте нет.
angle = nn.Parameter(torch.tensor(0.2))
optimizer = torch.optim.Adam([angle], lr=0.05)
# После RY(theta) вероятности двух исходов: cos²(theta/2), sin²(theta/2).
for _ in range(150):
    probability_one = torch.sin(angle/2).square()
    loss = (probability_one-0.8).square()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Угол:', angle.item())
print('Вероятность исхода 1:', torch.sin(angle/2).square().item())
