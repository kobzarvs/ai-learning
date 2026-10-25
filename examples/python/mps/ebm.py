# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import torch
from torch import nn

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

# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
# Создаём тензор — числовой массив — из указанных значений.
center = nn.Parameter(torch.tensor(0.))

# Adam подбирает величину шага для каждого параметра по истории градиентов.
optimizer = torch.optim.Adam([center], lr=0.05)

# Создаём искусственные данные из нормального распределения со средним 0.
data = torch.randn(200)*0.2+1.5
for _ in range(100):
    loss = (data-center).square().mean()

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()
x = torch.randn(100)
for _ in range(80):
    # Явно включаем или отключаем вычисление градиентов для этих параметров.
    x.requires_grad_(True)

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    energy = 0.5*(x-center.detach()).square()

    # Получаем производную выхода по входу.
    # create_graph=True позволяет затем вычислить производную ещё раз.
    gradient, = torch.autograd.grad(energy.sum(), x)

    # Создаём случайный шум из нормального распределения той же формы, что исходный
    # тензор.
    x = (x-0.1*gradient+math.sqrt(0.02)*torch.randn_like(x)).detach()
print('Найденный центр:', center.item(), 'среднее образцов:', x.mean().item())
