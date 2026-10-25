# Python 3.14 · MLX 0.32.4
import mlx.core as mx

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

# Возводим каждый элемент в квадрат; так получают квадраты ошибок.
def loss_fn(x): return mx.square(x-2)

# Получаем функцию, возвращающую и значение ошибки, и её градиент.
value_and_grad = mx.value_and_grad(loss_fn)
for rate in [0.05, 0.4, 1.1]:
    parameter, losses = mx.array(5.), []
    for _ in range(20):
        loss, gradient = value_and_grad(parameter)
        parameter = parameter-rate*gradient

        # MLX откладывает расчёт до запроса результата.
        # Здесь действительно выполняем накопленные вычисления.
        mx.eval(parameter, loss)
        losses.append(loss.item())
    print("Шаг:", rate, "первые ошибки:", losses[:3], "последняя:", losses[-1])
    assert losses[-1] < losses[0] if rate < 1 else losses[-1] > losses[0]
