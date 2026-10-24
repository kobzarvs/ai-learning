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


# Создадим обычное облако точек и добавим одну точку далеко от остальных. Будем случайно разрезать плоскость и считать, сколько разрезов нужно, чтобы оставить выбранную точку одну. Проверим, действительно ли далёкую точку обычно проще отделить. Это идея поиска необычных объектов; нейросеть здесь не обучается.
x = torch.cat([torch.randn(100, 2)*0.3, torch.tensor([[5.,5.]])])
def depth(values, index, level=0):
    if len(values)<=1 or level>=30:
        return level
    feature = int(torch.randint(0, values.shape[1], (1,)))
    low, high = values[:, feature].min(), values[:, feature].max()
    if high == low:
        return level
    cut = low + torch.rand(())*(high-low)
    side = values[:, feature] < cut
    keep = side == side[index]
    new_index = int(keep[:index].sum())
    return depth(values[keep], new_index, level+1)
for index in [0, 100]:
    paths = [depth(x, index) for _ in range(100)]
    print('Точка', index, 'средняя глубина:', sum(paths)/len(paths))
