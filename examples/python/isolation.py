# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Сравнить, сколько случайных разрезов требуется, чтобы изолировать обычную и далёкую точку.
x = torch.cat([torch.randn(100, 2)*0.3, torch.tensor([[5.,5.]])])
def depth(values, index, level=0):
    if len(values)<=1 or level>=30:
        return level
    feature = int(torch.randint(0, values.shape[1], (1,)))
    low, high = values[:, feature].min(), values[:, feature].max()
    if high == low:
        return level
    cut = low + torch.rand(())*(high-low)
    side = values[:, feature] < cut
    keep = side == side[index]
    new_index = int(keep[:index].sum())
    return depth(values[keep], new_index, level+1)
for index in [0, 100]:
    paths = [depth(x, index) for _ in range(100)]
    print('Точка', index, 'средняя глубина:', sum(paths)/len(paths))
