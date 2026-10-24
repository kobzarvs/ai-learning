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


# Возьмём большую матрицу (прямоугольную таблицу чисел), которая задаёт вычисление модели. Попробуем приблизительно заменить её двумя более узкими таблицами, выполняемыми по очереди. Чем уже промежуточная часть, тем меньше чисел нужно хранить, но тем больше информации можно потерять. Сравним число сохранённых значений и ошибку восстановления.
weights = torch.randn(64, 4)@torch.randn(4, 32)+0.01*torch.randn(64, 32)
u, s, vh = torch.linalg.svd(weights, full_matrices=False)
rank = 4
left = u[:, :rank]*s[:rank]
right = vh[:rank]
x = torch.randn(20, 32)
print('Ошибка:', F.mse_loss((x@right.T)@left.T, x@weights.T).item())
print('Параметров до и после:', weights.numel(), left.numel()+right.numel())
