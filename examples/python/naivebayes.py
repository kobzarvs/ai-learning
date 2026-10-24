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


# Создадим объекты двух категорий, каждый из которых описан несколькими числами — признаками. По учебным примерам оценим, какие значения каждого признака характерны для каждой категории. Затем по этим оценкам будем выбирать категорию нового объекта. Метод упрощает задачу: рассматривает признаки отдельно, не учитывая связи между ними внутри категории.
x = torch.cat([torch.randn(120, 2)-1, torch.randn(120, 2)+1])
y = torch.cat([torch.zeros(120), torch.ones(120)]).long()
order = torch.randperm(len(x)); x, y = x[order], y[order]
means, variances, priors = [], [], []
for label in range(2):
    subset = x[:180][y[:180] == label]
    means.append(subset.mean(0)); variances.append(subset.var(0)+1e-4)
    priors.append(len(subset)/180)
mu, var = torch.stack(means), torch.stack(variances)
log_score = -0.5*((x[180:, None]-mu).square()/var+var.log()).sum(2)
log_score += torch.tensor(priors).log()
print('Верно:', log_score.argmax(1).eq(y[180:]).float().mean().item())
