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
truth = torch.linspace(0, 2, 60)

# Создаём искусственные данные из нормального распределения со средним 0.
observed = truth+0.35*torch.randn(60)

# Создаём тензор — числовой массив — из указанных значений.
mean, variance, estimates = torch.tensor(0.), torch.tensor(1.), []
for measurement in observed:
    # Допускаем изменение состояния между шагами.
    variance = variance+0.01
    gain = variance/(variance+0.35**2)
    mean = mean+gain*(measurement-mean)
    variance = (1-gain)*variance
    estimates.append(mean)

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
estimates = torch.stack(estimates)

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка измерений:', F.mse_loss(observed, truth).item())
print('Ошибка фильтра:', F.mse_loss(estimates, truth).item())
