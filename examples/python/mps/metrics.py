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

# Создаём тензор — числовой массив — из указанных значений.
y = torch.tensor([0]*95+[1]*5)
prediction = torch.zeros_like(y)
true_positive = ((prediction==1)&(y==1)).sum()
false_positive = ((prediction==1)&(y==0)).sum()
false_negative = ((prediction==0)&(y==1)).sum()
accuracy_value = prediction.eq(y).float().mean()

# Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).
precision = true_positive/(true_positive+false_positive).clamp_min(1)
recall = true_positive/(true_positive+false_negative).clamp_min(1)
print('Общая точность:', accuracy_value.item())
print('Точность тревог:', precision.item(), 'доля найденных событий:', recall.item())
