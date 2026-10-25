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
x = torch.tensor(3., requires_grad=True)
y = x.square()+2*x

# Вычисляем градиенты ошибки по обучаемым параметрам.
y.backward()
h = 0.001

# Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
finite_difference = ((x.detach()+h).square()+2*(x.detach()+h)-y.detach())/h
print('Производная PyTorch:', x.grad.item())
print('Численная оценка:', finite_difference.item())
assert abs(x.grad.item()-8)<1e-6
