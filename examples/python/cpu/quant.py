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

# Создаём искусственные данные из нормального распределения со средним 0.
weights = torch.randn(128, 64)*0.1

# Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).
scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)/127
packed = (weights/scale).round().clamp(-127, 127).to(torch.int8)
restored = packed.float()*scale
x = torch.randn(20, 64)

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка выхода:', F.mse_loss(x@restored.T, x@weights.T).item())
print('Байт до:', weights.numel()*weights.element_size())
print('Байт после с масштабами:', packed.numel()*packed.element_size()+scale.numel()*scale.element_size())
