# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Ответить на вопрос по цепочке явных связей в графе знаний.
facts = [('камера', 'часть', 'робот'), ('робот', 'находится', 'цех'),
         ('датчик', 'часть', 'станок'), ('станок', 'находится', 'склад')]
object_name = 'камера'
parents = [b for a, relation, b in facts if a==object_name and relation=='часть']
places = [b for a, relation, b in facts if a in parents and relation=='находится']
print('Где находится камера:', places)
print('Основание:', [f for f in facts if f[0] in [object_name]+parents])
assert places == ['цех']
