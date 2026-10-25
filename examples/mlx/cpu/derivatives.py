# Python 3.14 · MLX 0.32.4
import mlx.core as mx

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

# Возводим каждый элемент в квадрат; так получают квадраты ошибок.
def f(x): return mx.square(x)+2*x
x = mx.array(3.)

# Создаём функцию, вычисляющую производную исходной функции по её аргументу.
derivative = mx.grad(f)(x).item()
numerical = ((f(x+0.001)-f(x))/0.001).item()
print("Производная MLX:", derivative, "численная оценка:", numerical)
assert abs(derivative-8) < 1e-6 and abs(numerical-8) < 0.01
