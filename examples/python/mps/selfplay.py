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
payoff = torch.tensor([[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]])
counts = torch.tensor([9., 3., 1.])
for _ in range(2000):
    opponent = counts/counts.sum()

    # Лучший ответ на среднюю стратегию собственных прошлых игр.
    action = int((payoff@opponent).argmax())
    counts[action] += 1
average = counts/counts.sum()
print('Средняя стратегия:', average.tolist())
print('Выгода лучшего ответа соперника:', (payoff@average).max().item())
assert (payoff@average).max().item() < 0.05
