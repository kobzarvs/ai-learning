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
def loss_fn(x): return mx.square(x-2)

# Получаем функцию, возвращающую и значение ошибки, и её градиент.
value_and_grad = mx.value_and_grad(loss_fn)
for rate in [0.05, 0.4, 1.1]:
    parameter, losses = mx.array(5.), []
    for _ in range(20):
        loss, gradient = value_and_grad(parameter)
        parameter = parameter-rate*gradient

        # MLX откладывает расчёт до запроса результата.
        # Здесь действительно выполняем накопленные вычисления.
        mx.eval(parameter, loss)
        losses.append(loss.item())
    print("Шаг:", rate, "первые ошибки:", losses[:3], "последняя:", losses[-1])
    assert losses[-1] < losses[0] if rate < 1 else losses[-1] > losses[0]
