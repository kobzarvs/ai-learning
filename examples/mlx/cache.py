# Python 3.14 · MLX 0.32.4
import argparse
import platform
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

parser = argparse.ArgumentParser(description="AI Atlas / MLX")
parser.add_argument("--device", choices=["cpu", "metal", "auto"], default="cpu")
requested = parser.parse_args().device
metal_available = platform.system() == "Darwin" and mx.device_count(mx.gpu) > 0
if requested == "auto":
    requested = "metal" if metal_available else "cpu"
if requested == "metal" and not metal_available:
    parser.error("Metal недоступен. Запусти нативно на Apple silicon вне Linux-контейнера либо явно выбери --device cpu.")
mx.set_default_device(mx.gpu if requested == "metal" else mx.cpu)
mx.random.seed(7)
print("MLX | устройство:", requested, "| dtype: float32")

def attention(q, k, v, causal=False):
    # Меняем две оси местами, чтобы согласовать формы перед матричным умножением.
    scores = (q@mx.swapaxes(k, -1, -2))/(q.shape[-1]**0.5)
    if causal:
        mask = mx.arange(k.shape[-2])[None, :] > mx.arange(q.shape[-2])[:, None]

        # По логической маске выбираем значение из одного из двух массивов.
        scores = mx.where(mask, -1e9, scores)

    # Преобразуем оценки в неотрицательные веса, сумма которых равна 1.
    return mx.softmax(scores, axis=-1)@v

# Создаём искусственные данные из нормального распределения со средним 0.
x = mx.random.normal((1, 1, 10, 8))
q, k, v = x+0.1, x-0.1, x*0.5
full = attention(q, k, v, True)
keys, values, outputs = [], [], []
for t in range(10):
    keys.append(k[:, :, t:t+1])
    values.append(v[:, :, t:t+1])

    # Соединяем массивы вдоль уже существующей оси.
    outputs.append(attention(q[:, :, t:t+1], mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)))
cached = mx.concatenate(outputs, axis=2)
print("Совпадает:", mx.allclose(full, cached, atol=1e-5).item())
assert mx.allclose(full, cached, atol=1e-5).item()
