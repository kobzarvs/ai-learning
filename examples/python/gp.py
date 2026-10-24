# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_DEVICES = ['cpu', 'cuda']
DEVICE_NOTE = 'Здесь используются линейная алгебра, комплексные FFT или специальные распределения; вариант MPS пока не включён.'

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


# Пусть известны значения некоторой кривой лишь в нескольких точках. Построим прогноз для промежутков между ними и для мест, где наблюдений нет. Вместе с прогнозом получим оценку разброса возможных значений (насколько модель не уверена в ответе). Проверим, где эта неопределённость оказывается больше.
x = torch.tensor([[-1.], [0.], [1.]])
y = x.sin()
query = torch.linspace(-3, 3, 21)[:, None]
def kernel(a, b):
    return torch.exp(-0.5*torch.cdist(a, b).square())
k = kernel(x, x)+0.01*torch.eye(len(x))
cross = kernel(query, x)
mean = cross @ torch.linalg.solve(k, y)
variance = (1-(cross*torch.linalg.solve(k, cross.T).T).sum(1)).clamp_min(0)
for i in [0, 10, 20]:
    print('x =', query[i].item(), 'среднее =', mean[i].item(), 'разброс =', variance[i].sqrt().item())
