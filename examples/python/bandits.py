# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_DEVICES = ['cpu', 'cuda']
DEVICE_NOTE = 'Здесь используются линейная алгебра, комплексные FFT или специальные распределения; вариант MPS пока не включён.'

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


# Представь три кнопки с неизвестными шансами выигрыша. Программа будет нажимать кнопки, получать успех или неудачу и постепенно пересчитывать оценки их полезности. Нужно сочетать проверку неизвестных вариантов с использованием уже удачных. Посмотрим, станет ли она чаще выбирать кнопку с настоящей вероятностью успеха 80%.
true_chance = torch.tensor([0.2, 0.5, 0.8])
success = torch.ones(3); failure = torch.ones(3)
count = torch.zeros(3, dtype=torch.long)
for _ in range(800):
    guesses = torch.distributions.Beta(success, failure).sample()
    action = int(guesses.argmax())
    reward = int(torch.rand(()) < true_chance[action])
    success[action] += reward; failure[action] += 1-reward
    count[action] += 1
print('Сколько раз выбрано каждое действие:', count.tolist())
print('Оценки вероятностей:', (success/(success+failure)).tolist())
