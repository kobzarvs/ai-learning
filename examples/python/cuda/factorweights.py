# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна: проверь видеокарту NVIDIA и установку PyTorch.")

device = torch.device("cuda")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

# Создаём искусственные данные из нормального распределения со средним 0.
weights = torch.randn(64, 4)@torch.randn(4, 32)+0.01*torch.randn(64, 32)

# Сингулярное разложение (SVD) выделяет направления матрицы и их значимость.
# Крупнейшие сингулярные числа позволяют построить компактное приближение.
u, s, vh = torch.linalg.svd(weights, full_matrices=False)
rank = 4
left = u[:, :rank]*s[:rank]
right = vh[:rank]
x = torch.randn(20, 32)

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка:', F.mse_loss((x@right.T)@left.T, x@weights.T).item())
print('Параметров до и после:', weights.numel(), left.numel()+right.numel())
