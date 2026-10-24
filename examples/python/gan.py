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


# Будем получать числа, похожие на примеры со средним около 2. Одна сеть, генератор, создаёт числа; другая, критик, учится отличать их от учебных примеров. Они по очереди улучшают свою работу. Проверим не только среднее созданных чисел, но и их разброс, чтобы генератор не отвечал одним и тем же числом.
generator = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
critic = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
g_opt = torch.optim.AdamW(generator.parameters(), lr=0.01)
d_opt = torch.optim.AdamW(critic.parameters(), lr=0.01)
for _ in range(300):
    real = torch.randn(64, 1)*0.4+2
    fake = generator(torch.randn(64, 1))
    d_loss = F.binary_cross_entropy_with_logits(critic(real), torch.ones_like(real))
    d_loss += F.binary_cross_entropy_with_logits(critic(fake.detach()), torch.zeros_like(fake))
    d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
    # Генератор получает градиент через критика, но не меняет его веса.
    critic.requires_grad_(False)
    g_loss = F.binary_cross_entropy_with_logits(critic(fake), torch.ones_like(fake))
    g_opt.zero_grad(set_to_none=True); g_loss.backward(); g_opt.step()
    critic.requires_grad_(True)
with torch.no_grad():
    generated = generator(torch.randn(1000, 1))
print('Среднее и разброс:', generated.mean().item(), generated.std().item())
