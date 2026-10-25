# Python 3.14 · MLX 0.32.4
import mlx.core as mx

if mx.device_count(mx.gpu) == 0:
    raise RuntimeError("Metal недоступен: нужен нативный запуск на Apple silicon.")

mx.set_default_device(mx.gpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: Metal")

x = mx.array([[1., 2., 3.], [4., 5., 6.]])
w = mx.array([[1., 0.], [0., 1.], [1., 1.]])
answer = x@w
print("Ответ:", answer.shape, answer.tolist())
assert answer.tolist() == [[4., 5.], [10., 11.]]
