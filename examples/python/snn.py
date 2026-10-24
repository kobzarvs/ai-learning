# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Проследить накопление напряжения и появление импульсов в модели нейрона.
current = torch.tensor([0.4]*20)
voltage, spikes = torch.tensor(0.), []
for value in current:
    voltage = 0.9*voltage+value
    spike = (voltage>=1).float()
    voltage = voltage-spike
    spikes.append(int(spike))
print('Импульсы:', spikes)
print('Всего импульсов:', sum(spikes))
