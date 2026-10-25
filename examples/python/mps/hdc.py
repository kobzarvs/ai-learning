# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

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

x, y = points(300)
projection = torch.randn(2, 1024)

# Для каждого элемента выбираем одно из двух значений по логическому условию.
codes = torch.where(x@projection>=0, 1., -1.)

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
prototypes = torch.stack([codes[:240][y[:240]==k].sum(0) for k in range(2)])

# Нормируем векторы; по умолчанию их евклидова длина становится равной 1.
prototypes = F.normalize(prototypes, dim=1)
prediction = (F.normalize(codes[240:], dim=1)@prototypes.T).argmax(1)
print('Доля верных ответов:', prediction.eq(y[240:]).float().mean().item())
