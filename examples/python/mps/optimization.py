# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
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

for learning_rate in [0.05, 0.4, 1.1]:
    # Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
    # Создаём тензор — числовой массив — из указанных значений.
    parameter = nn.Parameter(torch.tensor(5.))

    # SGD — градиентный спуск; параметр lr задаёт размер шага уменьшения ошибки.
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    losses = []
    for _ in range(20):
        loss = (parameter-2).square()

        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)

        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()

        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()

        # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
        losses.append(float(loss.detach()))
    print('Шаг:', learning_rate, 'первые ошибки:', losses[:3], 'последняя:', losses[-1])
