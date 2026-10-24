# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
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

# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Linear(20, 10)
# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(30, 20)
# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    before = model(x)
    # Находим пороги, ниже которых лежат заданные доли значений.
    threshold = torch.quantile(model.weight.abs(), 0.5)
    model.weight.mul_(model.weight.abs()>=threshold)
    after = model(x)
print('Нулевых весов:', int((model.weight==0).sum()))
print('Плотная память весов в байтах:', model.weight.numel()*model.weight.element_size())
# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка после удаления:', F.mse_loss(after, before).item())
