# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Проверить, что сохранённые ключи и значения дают тот же attention при чтении по одному токену.
x = torch.randn(1, 1, 10, 8)
q, k, v = x+0.1, x-0.1, x*0.5
full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
k_cache, v_cache, outputs = [], [], []
for t in range(10):
    k_cache.append(k[:, :, t:t+1]); v_cache.append(v[:, :, t:t+1])
    # У запроса длины 1 все сохранённые позиции уже в прошлом.
    output = F.scaled_dot_product_attention(q[:, :, t:t+1],
        torch.cat(k_cache, dim=2), torch.cat(v_cache, dim=2), is_causal=False)
    outputs.append(output)
cached = torch.cat(outputs, dim=2)
print('Совпадает с полным расчётом:', torch.allclose(full, cached, atol=1e-6))
assert torch.allclose(full, cached, atol=1e-6)
