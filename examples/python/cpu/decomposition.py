# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

# Создаём равномерную сетку между двумя границами, включая обе границы.
t = torch.linspace(-2, 2, 120)

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.stack([t, 2*t, -t], dim=1)+0.05*torch.randn(120, 3)
mean = x[:90].mean(0)

# Сингулярное разложение (SVD) выделяет направления матрицы и их значимость.
# Крупнейшие сингулярные числа позволяют построить компактное приближение.
u, s, vh = torch.linalg.svd(x[:90]-mean, full_matrices=False)
direction = vh[:1]
compressed = (x[90:]-mean) @ direction.T
restored = compressed @ direction + mean
print('Размер до и после:', tuple(x[90:].shape), tuple(compressed.shape))

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка восстановления:', F.mse_loss(restored, x[90:]).item())
