# Python 3.14 · MLX 0.32.4
import mlx.core as mx

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

signal = mx.sin(mx.arange(260)*0.13)

# Создаём искусственные данные из нормального распределения со средним 0.
w, input_weights = mx.random.normal((24,24))*0.08, mx.random.normal((24,))*0.3
state, states = mx.zeros((24,)), []
for value in signal[:-1]:
    # Ограничиваем каждое значение диапазоном от -1 до 1.
    state = mx.tanh(w@state+input_weights*value)
    states.append(state)

# Объединяем массивы одинаковой формы, добавляя новую ось.
x, y = mx.stack(states), signal[1:, None]
a, b = x[20:200], y[20:200]

# Решаем систему A @ x = b без явного вычисления обратной матрицы.
readout = mx.linalg.solve(a.T@a+0.001*mx.eye(24), a.T@b, stream=mx.cpu)

# Усредняем значения вдоль указанной оси или по всему массиву.
# Возводим каждый элемент в квадрат; так получают квадраты ошибок.
error = mx.mean(mx.square(x[200:]@readout-y[200:])).item()
print("Ошибка на более позднем участке:", error)
assert error < 0.02
