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


# Возьмём игру «камень, ножницы, бумага». На каждом шаге программа будет выбирать ход против того, как она сама в среднем играла раньше. Будем накапливать историю и смотреть, становятся ли три хода примерно одинаково частыми. Нейросети в этом примере нет: он показывает идею обучения на собственном прошлом поведении.
payoff = torch.tensor([[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]])
counts = torch.tensor([9., 3., 1.])
for _ in range(2000):
    opponent = counts/counts.sum()
    # Лучший ответ на среднюю стратегию собственных прошлых игр.
    action = int((payoff@opponent).argmax())
    counts[action] += 1
average = counts/counts.sum()
print('Средняя стратегия:', average.tolist())
print('Выгода лучшего ответа соперника:', (payoff@average).max().item())
assert (payoff@average).max().item() < 0.05
