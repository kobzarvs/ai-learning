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

observations = mx.array([1,1,0,1,1,0,1,1])

# Суммируем значения вдоль выбранной оси или по всему массиву.
a = 1+int(mx.sum(observations).item())
b = 1+len(observations)-(a-1)

# Генерируем значения равномерно внутри указанного интервала.
ga = -mx.sum(mx.log(mx.random.uniform(1e-7, 1, (10000, a))), axis=1)
gb = -mx.sum(mx.log(mx.random.uniform(1e-7, 1, (10000, b))), axis=1)

# Упорядочиваем значения по возрастанию.
samples = mx.sort(ga/(ga+gb))
low, high = samples[249].item(), samples[9749].item()
print("Оценка вероятности:", a/(a+b), "95% интервал:", [low, high])
assert 0.3 < low < 0.7 < high < 0.99
