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

weights = mx.random.normal((64,4))@mx.random.normal((4,32))+0.01*mx.random.normal((64,32))
# SVD выполняется явно на CPU; последующие умножения — на выбранном устройстве.
u, s, vt = mx.linalg.svd(weights, stream=mx.cpu)
left, right = u[:, :4]*s[:4], vt[:4]
x = mx.random.normal((20,32))
error = mx.mean(mx.square((x@right.T)@left.T-x@weights.T)).item()
print("Ошибка:", error, "параметров:", weights.size, "->", left.size+right.size)
assert error < 0.02
