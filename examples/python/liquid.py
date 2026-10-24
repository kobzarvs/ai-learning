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

class ContinuousCell(nn.Module):
    def __init__(self):
        super().__init__()
        # Линейный слой учится преобразованию y = x @ W.T + b.
        self.target = nn.Linear(2, 4)
        self.speed = nn.Linear(2, 4)
    def forward(self, state, x, dt):
        target = self.target(x).tanh()
        # Softplus плавно преобразует число в положительное значение.
        decay = torch.exp(-F.softplus(self.speed(x))*dt)
        return decay*state + (1-decay)*target
cell = ContinuousCell()
# Создаём тензор заданной формы и заполняем его единицами.
# Создаём тензор заданной формы и заполняем его нулями.
x, state = torch.ones(1, 2), torch.zeros(1, 4)
for dt in [0.01, 0.1, 1.0]:
    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    print('Интервал', dt, 'состояние', cell(state, x, dt).detach().tolist())
