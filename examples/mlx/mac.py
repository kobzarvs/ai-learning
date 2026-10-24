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

def fit(model, x, y, steps=160, classification=False, rate=0.02):
    def loss_fn(model, x, y):
        prediction = model(x)
        return (nn.losses.cross_entropy(prediction, y, reduction="mean") if classification
                else mx.mean(mx.square(prediction-y)))
    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=rate, bias_correction=True)
    for _ in range(steps):
        loss, grads = value_and_grad(model, x, y)
        optimizer.update(model, grads)
        # MLX вычисляет лениво: материализуем веса и состояние каждого шага.
        mx.eval(model.parameters(), optimizer.state, loss)
    return loss.item()

x = mx.random.normal((64,2))
y = x[:, :1]*2-x[:, 1:]
model = nn.Linear(2,1)
fit(model, x, y, steps=200)
mx.synchronize()
error = mx.mean(mx.square(model(x)-y)).item()
print("Ошибка:", error)
assert error < 0.05
