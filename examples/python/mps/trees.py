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
best = None
for feature in range(2):
    # Создаём равномерную сетку между двумя границами, включая обе границы.
    for threshold in torch.linspace(-2, 2, 30):
        left = x[:180, feature] < threshold
        if not left.any() or left.all():
            continue
        a = y[:180][left].float().mean() >= 0.5
        b = y[:180][~left].float().mean() >= 0.5

        # Для каждого элемента выбираем одно из двух значений по логическому условию.
        predicted = torch.where(left, a, b)
        error = (predicted != y[:180]).float().mean().item()
        if best is None or error < best[0]:
            best = (error, feature, threshold, a, b)
_, feature, threshold, a, b = best
predicted = torch.where(x[180:, feature] < threshold, a, b)
print('Признак:', feature, 'порог:', threshold.item())
print('Верно на новых строках:', (predicted == y[180:]).float().mean().item())
