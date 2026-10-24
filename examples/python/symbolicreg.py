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


# Создадим пары чисел по правилу y = 2x² + 0.5 (квадрат входного числа умножить на 2 и прибавить 0.5). Программа попробует три вида формул и подберёт их числовые коэффициенты. Сравним ошибки на новых входах и найдём формулу, которая лучше описывает данные. Это маленький перебор понятных человеку правил.
x = torch.linspace(-2, 2, 100)
y = 2*x.square()+0.5
candidates = {
    'a*x+b': lambda t: torch.stack([t, torch.ones_like(t)], 1),
    'a*x²+b': lambda t: torch.stack([t.square(), torch.ones_like(t)], 1),
    'a*sin(x)+b': lambda t: torch.stack([t.sin(), torch.ones_like(t)], 1),
}
for name, features in candidates.items():
    weights = torch.linalg.lstsq(features(x[:70]), y[:70]).solution
    error = F.mse_loss(features(x[70:])@weights, y[70:])
    print(name, 'коэффициенты', weights.tolist(), 'ошибка', error.item())
