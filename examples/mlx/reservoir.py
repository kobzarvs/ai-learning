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

signal = mx.sin(mx.arange(260)*0.13)
w, input_weights = mx.random.normal((24,24))*0.08, mx.random.normal((24,))*0.3
state, states = mx.zeros((24,)), []
for value in signal[:-1]:
    state = mx.tanh(w@state+input_weights*value)
    states.append(state)
x, y = mx.stack(states), signal[1:, None]
a, b = x[20:200], y[20:200]
readout = mx.linalg.solve(a.T@a+0.001*mx.eye(24), a.T@b, stream=mx.cpu)
error = mx.mean(mx.square(x[200:]@readout-y[200:])).item()
print("Ошибка на более позднем участке:", error)
assert error < 0.02
