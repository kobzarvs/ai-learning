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

state = mx.zeros((4,))
decay = mx.array([0.1, 0.5, 0.8, 0.95])
outputs = []
for t in range(30):
    state = decay*state+(1 if t == 3 else 0)
    # Усредняем значения вдоль указанной оси или по всему массиву.
    outputs.append(mx.mean(state).item())
print("Ответ после импульса:", outputs[3:12])
assert outputs[3] > outputs[8] > outputs[20] > 0
