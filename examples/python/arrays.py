# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Разобраться в форме тензора и увидеть, как пакет объектов проходит линейный слой.
x = torch.tensor([[1.,2.,3.], [4.,5.,6.]])
weights = torch.tensor([[1.,0.], [0.,1.], [1.,1.]])
answer = x@weights
print('Вход:', tuple(x.shape), 'веса:', tuple(weights.shape))
print('Ответ:', tuple(answer.shape), answer.tolist())
assert answer.tolist() == [[4.,5.], [10.,11.]]
