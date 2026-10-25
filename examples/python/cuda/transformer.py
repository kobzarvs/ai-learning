# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна: проверь видеокарту NVIDIA и установку PyTorch.")

device = torch.device("cuda")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

# 2 последовательности, 6 позиций, 16 признаков.
x = torch.randn(2, 6, 16)

# Линейный слой учится преобразованию y = x @ W.T + b.
projection = nn.Linear(16, 48, bias=False)
q, k, v = projection(x).chunk(3, dim=-1)

# Измерения: набор, голова, позиция, признаки головы.
q, k, v = [z.view(2, 6, 4, 4).transpose(1, 2) for z in (q, k, v)]

# Внимание взвешивает значения V по сходству запросов Q и ключей K.
answer = F.scaled_dot_product_attention(q, k, v, is_causal=True)
changed_v = v.clone()
changed_v[:, :, -1] += 100
changed = F.scaled_dot_product_attention(q, k, changed_v, is_causal=True)
print('Размер ответа:', tuple(answer.shape))

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
print('Ранние ответы не меняются:', torch.allclose(answer[:, :, :-1], changed[:, :, :-1]))
assert torch.allclose(answer[:, :, :-1], changed[:, :, :-1])
