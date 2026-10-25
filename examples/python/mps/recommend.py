# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
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

# Создаём искусственные данные из нормального распределения со средним 0.
true_users, true_items = torch.randn(12, 3), torch.randn(10, 3)
rating = true_users @ true_items.T

# Каждому целочисленному индексу сопоставляем обучаемый вектор.
users, items = nn.Embedding(12, 3), nn.Embedding(10, 3)

# Генерируем случайные числа, равномерно распределённые от 0 до 1.
mask = torch.rand(12, 10) < 0.8

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(list(users.parameters())+list(items.parameters()), lr=0.04)
for _ in range(400):
    predictions = users.weight @ items.weight.T

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(predictions[mask], rating[mask])

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    prediction = users.weight @ items.weight.T
print('Ошибка на скрытых оценках:', F.mse_loss(prediction[~mask], rating[~mask]).item())
