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


# Создадим искусственные наблюдения, где третья величина влияет и на выбранное действие, и на результат. Сравним обычную связь «действие чаще встречается вместе с результатом» с эффектом принудительного изменения действия при прочих равных. Это различие между совпадением в данных и причиной. Правило генерации нам известно, поэтому настоящий эффект можно проверить.
weather = torch.randn(10000)
ice_cream = weather+0.2*torch.randn(10000)
swimming = 2*weather+0.2*torch.randn(10000)
correlation = torch.corrcoef(torch.stack([ice_cream, swimming]))[0,1]
# В нашей причинной схеме купание зависит от погоды, не от мороженого.
intervention = torch.full_like(ice_cream, 100.)
swimming_after = 2*weather+0.2*torch.randn(10000)
print('Корреляция:', correlation.item())
print('Изменение среднего купания при вмешательстве:', (swimming_after-swimming).mean().item())
assert intervention.mean() > ice_cream.mean()
