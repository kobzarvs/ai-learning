# Python 3.14 · MLX 0.32.4
import mlx.core as mx

if mx.device_count(mx.gpu) == 0:
    raise RuntimeError("Metal недоступен: нужен нативный запуск на Apple silicon.")

mx.set_default_device(mx.gpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: Metal")

state = mx.zeros((4,))
decay = mx.array([0.1, 0.5, 0.8, 0.95])
outputs = []
for t in range(30):
    state = decay*state+(1 if t == 3 else 0)

    # Усредняем значения вдоль указанной оси или по всему массиву.
    outputs.append(mx.mean(state).item())
print("Ответ после импульса:", outputs[3:12])
assert outputs[3] > outputs[8] > outputs[20] > 0
