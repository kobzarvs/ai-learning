# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна: проверь видеокарту NVIDIA и установку PyTorch.")

device = torch.device("cuda")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Linear(20, 10)

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(30, 20)

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    before = model(x)

    # Находим пороги, ниже которых лежат заданные доли значений.
    threshold = torch.quantile(model.weight.abs(), 0.5)
    model.weight.mul_(model.weight.abs()>=threshold)
    after = model(x)
print('Нулевых весов:', int((model.weight==0).sum()))
print('Плотная память весов в байтах:', model.weight.numel()*model.weight.element_size())

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка после удаления:', F.mse_loss(after, before).item())
