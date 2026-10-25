# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS недоступен: нужен совместимый Mac и нативный запуск.")

device = torch.device("mps")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

def points(count=320):
    # Создаём искусственные данные из нормального распределения со средним 0.
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

x, y = points(240)

# Вычисляем расстояния между каждой парой объектов из двух наборов.
distance = torch.cdist(x[180:], x[:180])

# Находим k крайних значений и их индексы; largest=False выбирает наименьшие.
neighbors = distance.topk(3, largest=False).indices
prediction = y[:180][neighbors].float().mean(1).ge(0.5).long()
print('Верно на новых точках:', prediction.eq(y[180:]).float().mean().item())
print('Соседи первой новой точки:', neighbors[0].tolist())
