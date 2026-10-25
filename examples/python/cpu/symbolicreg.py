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
x = torch.linspace(-2, 2, 100)
y = 2*x.square()+0.5

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
candidates = {
    'a*x+b': lambda t: torch.stack([t, torch.ones_like(t)], 1),
    'a*x²+b': lambda t: torch.stack([t.square(), torch.ones_like(t)], 1),
    'a*sin(x)+b': lambda t: torch.stack([t.sin(), torch.ones_like(t)], 1),
}
for name, features in candidates.items():
    # Подбираем коэффициенты, минимизирующие сумму квадратов невязок A @ x - b.
    weights = torch.linalg.lstsq(features(x[:70]), y[:70]).solution

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    error = F.mse_loss(features(x[70:])@weights, y[70:])
    print(name, 'коэффициенты', weights.tolist(), 'ошибка', error.item())
