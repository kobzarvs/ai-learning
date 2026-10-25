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
weights = torch.randn(32, 16)
scale = weights.abs().mean(1, keepdim=True)

# Для каждого элемента выбираем одно из двух значений по логическому условию.
binary = torch.where(weights>=0, 1., -1.)*scale
x = torch.randn(40, 16)

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка выхода:', F.mse_loss(x@binary.T, x@weights.T).item())
print('Теоретические биты знаков:', weights.numel())
print('Фактические байты этого float32-тензора:', binary.numel()*binary.element_size())
