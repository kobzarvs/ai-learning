# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Соединить вероятностный ответ сети с явным правилом допустимости.
# Модель предлагает действия: читать, отправить, удалить.
logits = torch.tensor([[1., 2., 5.]])
allowed = torch.tensor([[True, True, False]])
restricted = logits.masked_fill(~allowed, float('-inf'))
probabilities = restricted.softmax(1)
action = probabilities.argmax(1).item()
print('Вероятности допустимых действий:', probabilities.tolist())
print('Выбрано действие:', ['читать', 'отправить', 'удалить'][action])
assert probabilities[0, 2] == 0
