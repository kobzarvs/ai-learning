# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Передать сообщения между связанными объектами и увидеть, что изолированный объект не получает чужие сведения.
adjacency = torch.tensor([[1.,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]])
features = torch.tensor([[1.,0],[0,1],[1,1],[8,8]])
weights = adjacency / adjacency.sum(1, keepdim=True)
layer = nn.Linear(2, 3, bias=False)
messages = weights @ features
answer = layer(messages)
print('Сведения после обмена:', messages.tolist())
print('Изолированный объект:', messages[-1].tolist())
assert torch.allclose(messages[-1], features[-1])
