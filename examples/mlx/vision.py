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

# MLX использует NHWC: пакет, высота, ширина, каналы.
y = (mx.arange(240) % 2).astype(mx.int32)
positions = 1+(mx.arange(240)//2)%5
axis = mx.arange(8)
vertical = ((axis[None, :]-positions[:, None] >= 0) &
            (axis[None, :]-positions[:, None] < 2))[:, None, :]
horizontal = mx.transpose(vertical, (0, 2, 1))
x = mx.where(y[:, None, None] == 0, vertical, horizontal).astype(mx.float32)
x = x[:, :, :, None]+0.12*mx.random.normal((240, 8, 8, 1))
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.output = nn.Linear(8, 2)
    def __call__(self, x):
        return self.output(mx.mean(nn.silu(self.conv(x)), axis=(1, 2)))
model = CNN()
fit(model, x[:180], y[:180], classification=True)
accuracy = mx.mean(mx.argmax(model(x[180:]), axis=1) == y[180:]).item()
print("Верно на новых картинках:", accuracy)
print("Обучаемых чисел:", sum(p.size for _, p in tree_flatten(model.parameters())))
assert accuracy > 0.9
