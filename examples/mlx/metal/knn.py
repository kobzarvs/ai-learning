# Python 3.14 · MLX 0.32.4
import mlx.core as mx

if mx.device_count(mx.gpu) == 0:
    raise RuntimeError("Metal недоступен: нужен нативный запуск на Apple silicon.")

mx.set_default_device(mx.gpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: Metal")

# Создаём искусственные данные из нормального распределения со средним 0.
x = mx.random.normal((240, 2))
y = (x[:, 0]+0.5*x[:, 1] > 0).astype(mx.int32)

# Суммируем значения вдоль выбранной оси или по всему массиву.
# Возводим каждый элемент в квадрат; так получают квадраты ошибок.
distances = mx.sum(mx.square(x[180:, None, :]-x[None, :180, :]), axis=2)

# Получаем индексы элементов в порядке возрастания их значений.
neighbors = mx.argsort(distances, axis=1)[:, :3]

# Усредняем значения вдоль указанной оси или по всему массиву.
prediction = mx.mean(y[neighbors], axis=1) >= 0.5
accuracy = mx.mean(prediction == y[180:]).item()
print("Верно на новых точках:", accuracy, "соседи:", neighbors[0].tolist())
assert accuracy > 0.8
