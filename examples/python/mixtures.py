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


# Создадим числа, полученные из двух разных групп, и смешаем их. Программа будет поочерёдно оценивать центры и разброс групп, а затем вероятность принадлежности каждого числа каждой группе. Вероятность здесь нужна потому, что группы могут пересекаться и однозначный ответ не всегда возможен. Проверим найденные центры и распределение примеров между группами.
x = torch.cat([torch.randn(120)*0.3-1.5, torch.randn(120)*0.3+1.5])
means = torch.tensor([-1., 1.]); variance = torch.ones(2); weights = torch.ones(2)/2
for _ in range(30):
    score = -0.5*((x[:, None]-means).square()/variance+variance.log())+weights.log()
    responsibility = score.softmax(1)
    count = responsibility.sum(0)
    means = (responsibility*x[:, None]).sum(0)/count
    variance = (responsibility*(x[:, None]-means).square()).sum(0)/count
    variance = variance.clamp_min(1e-4)
    weights = count/len(x)
print('Центры:', means.tolist(), 'доли:', weights.tolist())
