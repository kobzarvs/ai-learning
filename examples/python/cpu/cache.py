# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(1, 1, 10, 8)
q, k, v = x+0.1, x-0.1, x*0.5

# Внимание взвешивает значения V по сходству запросов Q и ключей K.
full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
k_cache, v_cache, outputs = [], [], []
for t in range(10):
    k_cache.append(k[:, :, t:t+1])
    v_cache.append(v[:, :, t:t+1])

    # У запроса длины 1 все сохранённые позиции уже в прошлом.
    output = F.scaled_dot_product_attention(q[:, :, t:t+1],
        torch.cat(k_cache, dim=2), torch.cat(v_cache, dim=2), is_causal=False)
    outputs.append(output)
cached = torch.cat(outputs, dim=2)

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
print('Совпадает с полным расчётом:', torch.allclose(full, cached, atol=1e-6))
assert torch.allclose(full, cached, atol=1e-6)
