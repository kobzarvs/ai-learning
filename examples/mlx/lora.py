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

x = mx.random.normal((300, 8))
base = mx.random.normal((8, 8))*0.2
base_before = mx.array(base)
true_update = mx.random.normal((8, 2))@mx.random.normal((2, 8))*0.1
y = x@(base+true_update)
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = mx.random.normal((8, 2))*0.01
        self.b = mx.zeros((2, 8))
    def __call__(self, x):
        # База находится вне Module и не передаётся оптимизатору.
        return x@base+x@self.a@self.b
model = Adapter()
before = mx.mean(mx.square(model(x[240:])-y[240:])).item()
fit(model, x[:240], y[:240], steps=250, rate=0.04)
after = mx.mean(mx.square(model(x[240:])-y[240:])).item()
merged = base+model.a@model.b
print("MSE до:", before, "после:", after, "базовых весов:", base.size, "обучаемых:", model.a.size+model.b.size)
assert after < before*0.02 and after < 0.001
assert mx.array_equal(base, base_before).item()
assert mx.allclose(model(x[240:]), x[240:]@merged, atol=1e-5).item()
