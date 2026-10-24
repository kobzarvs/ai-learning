# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Посмотреть, как attention связывает элементы последовательности, и проверить запрет смотреть в будущее.
x = torch.randn(2, 6, 16)  # 2 последовательности, 6 позиций, 16 признаков.
projection = nn.Linear(16, 48, bias=False)
q, k, v = projection(x).chunk(3, dim=-1)
# Измерения: набор, голова, позиция, признаки головы.
q, k, v = [z.view(2, 6, 4, 4).transpose(1, 2) for z in (q, k, v)]
answer = F.scaled_dot_product_attention(q, k, v, is_causal=True)
changed_v = v.clone()
changed_v[:, :, -1] += 100
changed = F.scaled_dot_product_attention(q, k, changed_v, is_causal=True)
print('Размер ответа:', tuple(answer.shape))
print('Ранние ответы не меняются:', torch.allclose(answer[:, :, :-1], changed[:, :, :-1]))
assert torch.allclose(answer[:, :, :-1], changed[:, :, :-1])
