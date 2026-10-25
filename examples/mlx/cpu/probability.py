# Python 3.14 · MLX 0.32.4
import mlx.core as mx

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

observations = mx.array([1,1,0,1,1,0,1,1])

# Суммируем значения вдоль выбранной оси или по всему массиву.
a = 1+int(mx.sum(observations).item())
b = 1+len(observations)-(a-1)

# Генерируем значения равномерно внутри указанного интервала.
ga = -mx.sum(mx.log(mx.random.uniform(1e-7, 1, (10000, a))), axis=1)
gb = -mx.sum(mx.log(mx.random.uniform(1e-7, 1, (10000, b))), axis=1)

# Упорядочиваем значения по возрастанию.
samples = mx.sort(ga/(ga+gb))
low, high = samples[249].item(), samples[9749].item()
print("Оценка вероятности:", a/(a+b), "95% интервал:", [low, high])
assert 0.3 < low < 0.7 < high < 0.99
