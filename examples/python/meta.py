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


def fit_regression(model, x, y, steps=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

# Будем учить модель сразу на семействе похожих задач: у каждой своё правило «умножить вход на число и прибавить другое число». После каждой задачи скорректируем общие начальные настройки. Затем дадим новое правило и всего пять примеров для быстрой подстройки. Проверим, насколько уменьшится ошибка после десяти шагов обучения.
initial = nn.Linear(1, 1)
for _ in range(120):
    task = copy.deepcopy(initial)
    x = torch.randn(12, 1)
    slope = 1+torch.rand(()); bias = torch.rand(())*0.4
    fit_regression(task, x, slope*x+bias, steps=10)
    with torch.no_grad():
        for shared, adapted in zip(initial.parameters(), task.parameters()):
            shared.lerp_(adapted, 0.15)
x = torch.randn(30, 1); y = 1.7*x+0.15
adapted = copy.deepcopy(initial)
fit_regression(adapted, x[:5], y[:5], steps=10)
with torch.no_grad():
    print('До адаптации:', F.mse_loss(initial(x[5:]), y[5:]).item())
    print('После 10 шагов:', F.mse_loss(adapted(x[5:]), y[5:]).item())
