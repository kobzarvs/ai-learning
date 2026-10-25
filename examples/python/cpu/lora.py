# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
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
x = torch.randn(300, 8)
base = torch.randn(8, 8)*0.2
true_update = torch.randn(8, 2)@torch.randn(2, 8)*0.1
y = x@(base+true_update)

# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
a = nn.Parameter(torch.randn(8, 2)*0.01)

# Создаём тензор заданной формы и заполняем его нулями.
b = nn.Parameter(torch.zeros(2, 8))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW([a, b], lr=0.04)
for _ in range(250):
    prediction = x[:240]@base + x[:240]@a@b

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(prediction, y[:240])

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    print('Ошибка на новых объектах:', F.mse_loss(x[240:]@base+x[240:]@a@b, y[240:]).item())
print('Весов основной матрицы:', base.numel(), 'обучаемых:', a.numel()+b.numel())
