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

# Создадим объекты, описанные четырьмя связанными числами. Научим сеть сжимать их до одного числа, а затем восстанавливать исходные четыре: такой способ обучения называется автоэнкодером. После этого сравним ошибку восстановления обычного объекта и объекта, нарушающего привычную связь. Большая ошибка будет поводом обратить на объект внимание, но сама по себе не докажет неисправность.
normal = torch.randn(300, 1) @ torch.tensor([[1.,2.,-1.,0.5]])
normal += 0.03*torch.randn_like(normal)
model = nn.Sequential(nn.Linear(4, 1), nn.Linear(1, 4))
fit_regression(model, normal[:240], normal[:240], steps=300)
with torch.no_grad():
    ordinary = (model(normal[240:])-normal[240:]).square().mean(1)
    strange = torch.tensor([[5.,-5.,5.,-5.]])
    unusual = (model(strange)-strange).square().mean()
print('Обычные: средняя ошибка', ordinary.mean().item())
print('Необычный: ошибка', unusual.item())
