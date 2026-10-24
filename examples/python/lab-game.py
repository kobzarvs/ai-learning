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


# Создадим коридор из пяти клеток: можно двигаться влево и вправо, а награда находится в правой крайней клетке. За лишние шаги даётся небольшой штраф. Научим сеть оценивать пользу каждого действия с учётом возможных следующих шагов. Проверим, выберет ли она движение вправо из остальных четырёх клеток.
model = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
# Собираем все переходы маленькой среды. Действие 0 — влево, 1 — вправо.
states, actions, next_states, rewards, done = [], [], [], [], []
for state in range(4):
    for action in range(2):
        nxt = max(0, min(4, state+(1 if action else -1)))
        states.append(state); actions.append(action); next_states.append(nxt)
        rewards.append(1.0 if nxt==4 else -0.02); done.append(nxt==4)
s = F.one_hot(torch.tensor(states), 5).float()
ns = F.one_hot(torch.tensor(next_states), 5).float()
a = torch.tensor(actions); reward = torch.tensor(rewards); terminal = torch.tensor(done)
for _ in range(300):
    with torch.no_grad():
        target = reward+0.95*model(ns).max(1).values*(~terminal)
    value = model(s).gather(1, a[:, None]).squeeze(1)
    loss = F.mse_loss(value, target)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    print('Действия в клетках 0–3:', model(torch.eye(5)[:4]).argmax(1).tolist())
