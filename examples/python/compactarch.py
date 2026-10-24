# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Сравнить число параметров обычной свёртки и разделённой на обработку каналов и их смешивание.
x = torch.randn(1, 16, 24, 24)
regular = nn.Conv2d(16, 32, 3, padding=1, bias=False)
compact = nn.Sequential(
    nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
    nn.Conv2d(16, 32, 1, bias=False),
)
for name, model in [('Обычная', regular), ('Компактная', compact)]:
    count = sum(p.numel() for p in model.parameters())
    print(name, 'параметров:', count, 'байт весов:', count*4,
          'размер ответа:', tuple(model(x).shape))
