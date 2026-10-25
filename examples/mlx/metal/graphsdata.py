# Python 3.14 · MLX 0.32.4
import mlx.core as mx
import mlx.nn as nn

if mx.device_count(mx.gpu) == 0:
    raise RuntimeError("Metal недоступен: нужен нативный запуск на Apple silicon.")

mx.set_default_device(mx.gpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: Metal")

adjacency = mx.array([[1.,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]])
features = mx.array([[1.,0],[0,1],[1,1],[8,8]])

# Суммируем значения вдоль выбранной оси или по всему массиву.
messages = (adjacency/mx.sum(adjacency, axis=1, keepdims=True))@features

# Линейный слой учится преобразованию y = x @ W.T + b.
answer = nn.Linear(2, 3, bias=False)(messages)
print("Сведения после обмена:", messages.tolist(), "форма ответа:", answer.shape)
assert mx.allclose(messages[-1], features[-1]).item()
