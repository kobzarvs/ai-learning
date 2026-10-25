# Python 3.14 · MLX 0.32.4
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

def fit(model, x, y, steps=160, classification=False, rate=0.02):
    def loss_fn(model, x, y):
        prediction = model(x)
        if classification:
            # Измеряем ошибку классификации по исходным оценкам классов и верным меткам.
            return nn.losses.cross_entropy(prediction, y, reduction="mean")

        # Для численного прогноза сначала найдём ошибку каждого ответа.
        differences = prediction - y

        # Возводим каждый элемент в квадрат; так получают квадраты ошибок.
        squared_errors = mx.square(differences)

        # Усредняем значения вдоль указанной оси или по всему массиву.
        return mx.mean(squared_errors)

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

# MLX использует NHWC: пакет, высота, ширина, каналы.
y = (mx.arange(240) % 2).astype(mx.int32)
positions = 1+(mx.arange(240)//2)%5
axis = mx.arange(8)
vertical = ((axis[None, :]-positions[:, None] >= 0) &
            (axis[None, :]-positions[:, None] < 2))[:, None, :]

# Переставляем оси массива в указанный порядок.
horizontal = mx.transpose(vertical, (0, 2, 1))

# По логической маске выбираем значение из одного из двух массивов.
x = mx.where(y[:, None, None] == 0, vertical, horizontal).astype(mx.float32)

# Создаём искусственные данные из нормального распределения со средним 0.
x = x[:, :, :, None]+0.12*mx.random.normal((240, 8, 8, 1))
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Свёртка применяет обучаемые фильтры к небольшим участкам изображения.
        self.conv = nn.Conv2d(1, 8, 3, padding=1)

        # Линейный слой учится преобразованию y = x @ W.T + b.
        self.output = nn.Linear(8, 2)
    def __call__(self, x):
        return self.output(mx.mean(nn.silu(self.conv(x)), axis=(1, 2)))
model = CNN()
fit(model, x[:180], y[:180], classification=True)

# Находим индекс наибольшей оценки — выбранную моделью категорию.
accuracy = mx.mean(mx.argmax(model(x[180:]), axis=1) == y[180:]).item()
print("Верно на новых картинках:", accuracy)

# Разворачиваем дерево параметров модели в список пар «путь, массив».
print("Обучаемых чисел:", sum(p.size for _, p in tree_flatten(model.parameters())))
assert accuracy > 0.9
