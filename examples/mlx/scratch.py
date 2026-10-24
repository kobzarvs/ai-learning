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
        # Измеряем ошибку классификации по исходным оценкам классов и верным меткам.
        # Усредняем значения вдоль указанной оси или по всему массиву.
        return (nn.losses.cross_entropy(prediction, y, reduction="mean") if classification
                else mx.mean(mx.square(prediction-y)))
    # Вычисляем ошибку и градиенты по обучаемым параметрам модели MLX.
    value_and_grad = nn.value_and_grad(model, loss_fn)
    # Adam обновляет параметры, учитывая историю их градиентов.
    optimizer = optim.Adam(learning_rate=rate, bias_correction=True)
    for _ in range(steps):
        loss, grads = value_and_grad(model, x, y)
        # Применяем вычисленные градиенты к параметрам модели.
        optimizer.update(model, grads)
        # MLX откладывает расчёт до запроса результата.
        # Здесь выполняем накопленные вычисления весов, состояния и ошибки.
        mx.eval(model.parameters(), optimizer.state, loss)
    return loss.item()

# Создаём искусственные данные из нормального распределения со средним 0.
x = mx.random.normal((320, 2))
y = (x[:, 0]+0.5*x[:, 1] > 0).astype(mx.int32)
# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
# Находим индекс наибольшей оценки — выбранную моделью категорию.
print("До:", mx.mean(mx.argmax(model(x[240:]), axis=1) == y[240:]).item())
loss = fit(model, x[:240], y[:240], classification=True)
accuracy = mx.mean(mx.argmax(model(x[240:]), axis=1) == y[240:]).item()
print("После:", accuracy, "последняя учебная ошибка:", loss)
assert accuracy > 0.9
