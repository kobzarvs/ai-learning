# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Проверить, что линейное attention можно вычислять через накопленную память без таблицы всех пар позиций.
q, k, v = [torch.rand(12, 8) for _ in range(3)]
# Положительные признаки делают знаменатель устойчивым.
phi = lambda x: F.elu(x) + 1
q, k = phi(q), phi(k)
memory, normalizer = torch.zeros(8, 8), torch.zeros(8)
answers = []
for t in range(len(q)):
    memory += torch.outer(k[t], v[t])
    normalizer += k[t]
    answers.append((q[t] @ memory) / (q[t] @ normalizer + 1e-6))
recurrent = torch.stack(answers)
scores = (q @ k.T).tril()
reference = (scores @ v) / (scores.sum(1, keepdim=True) + 1e-6)
print('Максимальная разница:', (recurrent-reference).abs().max().item())
assert torch.allclose(recurrent, reference, atol=1e-5)
