# Python 3.14 · MLX 0.32.4
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

if mx.device_count(mx.gpu) == 0:
    raise RuntimeError("Metal недоступен: нужен нативный запуск на Apple silicon.")

mx.set_default_device(mx.gpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: Metal")

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

# Создаём искусственные данные из нормального распределения со средним 0.
normal = mx.random.normal((300, 1))@mx.array([[1., 2., -1., 0.5]])
normal = normal+0.03*mx.random.normal(normal.shape)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(4, 1), nn.Linear(1, 4))
fit(model, normal[:240], normal[:240], steps=300)
ordinary = mx.mean(mx.square(model(normal[240:])-normal[240:])).item()
strange = mx.array([[5., -5., 5., -5.]])
unusual = mx.mean(mx.square(model(strange)-strange)).item()
print("Обычные:", ordinary, "необычный:", unusual)
assert ordinary < 0.02 and unusual > ordinary*100
