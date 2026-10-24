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

# Возводим каждый элемент в квадрат; так получают квадраты ошибок.
def f(x): return mx.square(x)+2*x
x = mx.array(3.)

# Создаём функцию, вычисляющую производную исходной функции по её аргументу.
derivative = mx.grad(f)(x).item()
numerical = ((f(x+0.001)-f(x))/0.001).item()
print("Производная MLX:", derivative, "численная оценка:", numerical)
assert abs(derivative-8) < 1e-6 and abs(numerical-8) < 0.01
