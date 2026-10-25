# Python 3.14 · MLX 0.32.4
import mlx.core as mx

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

# Создаём искусственные данные из нормального распределения со средним 0.
weights = mx.random.normal((64,4))@mx.random.normal((4,32))+0.01*mx.random.normal((64,32))

# SVD выполняется явно на CPU; последующие умножения — на выбранном устройстве.
u, s, vt = mx.linalg.svd(weights, stream=mx.cpu)
left, right = u[:, :4]*s[:4], vt[:4]
x = mx.random.normal((20,32))

# Усредняем значения вдоль указанной оси или по всему массиву.
# Возводим каждый элемент в квадрат; так получают квадраты ошибок.
error = mx.mean(mx.square((x@right.T)@left.T-x@weights.T)).item()
print("Ошибка:", error, "параметров:", weights.size, "->", left.size+right.size)
assert error < 0.02
