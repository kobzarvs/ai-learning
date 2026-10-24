# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Направить каждый объект к одному из трёх маленьких экспертов.
x = torch.randn(10, 4)
router = nn.Linear(4, 3)
experts = nn.ModuleList([nn.Linear(4, 2) for _ in range(3)])
scores = router(x).softmax(dim=1)
choice = scores.argmax(dim=1)
answer = torch.zeros(10, 2)
for i, expert in enumerate(experts):
    selected = choice == i
    answer[selected] = expert(x[selected])
print('Эксперт для каждого объекта:', choice.tolist())
print('Объектов у экспертов:', torch.bincount(choice, minlength=3).tolist())
print('Ответов:', tuple(answer.shape))
