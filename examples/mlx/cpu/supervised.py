# Python 3.14 · MLX 0.32.4
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
