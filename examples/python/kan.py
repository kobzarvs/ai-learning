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

# На вход подаём два числа, а правильный ответ получаем по формуле sin(x₁) + x₂² (синус первого числа плюс квадрат второго). Вместо одного множителя для каждого входа модель учится подбирать целую кривую, затем складывает результаты. Проверим, сможет ли она повторить это правило на новых парах чисел. Так мы знакомимся с идеей обучаемых функций на связях сети.
x = torch.rand(400, 2) * 2 - 1
y = (x[:, :1].sin() + x[:, 1:].square())
class EdgeFunctions(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Buffer(torch.linspace(-1, 1, 12))
        self.coefficients = nn.Parameter(torch.zeros(2, 12))
    def forward(self, x):
        bases = torch.exp(-((x[..., None]-self.centers)/0.25).square())
        return (bases*self.coefficients).sum((1, 2)).unsqueeze(1)
model = EdgeFunctions()
fit_regression(model, x[:320], y[:320])
with torch.no_grad():
    print('Ошибка на новых точках:', F.mse_loss(model(x[320:]), y[320:]).item())
