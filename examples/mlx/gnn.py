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

adjacency = mx.array([[1.,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]])
features = mx.array([[1.,0],[0,1],[1,1],[8,8]])
# Суммируем значения вдоль выбранной оси или по всему массиву.
messages = (adjacency/mx.sum(adjacency, axis=1, keepdims=True))@features
# Линейный слой учится преобразованию y = x @ W.T + b.
answer = nn.Linear(2, 3, bias=False)(messages)
print("Сведения после обмена:", messages.tolist(), "форма ответа:", answer.shape)
assert mx.allclose(messages[-1], features[-1]).item()
