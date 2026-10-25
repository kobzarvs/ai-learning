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

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
velocity = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 2))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(velocity.parameters(), lr=0.01)
for _ in range(350):
    # Создаём искусственные данные из нормального распределения со средним 0.
    source = torch.randn(128, 2)

    # Создаём тензор — числовой массив — из указанных значений.
    target = torch.randn(128, 2)*0.3 + torch.tensor([2.,-1.])

    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    time = torch.rand(128, 1)
    position = (1-time)*source + time*target

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    # Соединяем тензоры вдоль уже существующего измерения dim.
    loss = F.mse_loss(velocity(torch.cat([position, time], 1)), target-source)

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    samples = torch.randn(500, 2)
    for step in range(60):
        time = torch.full((500, 1), step/60)
        samples += velocity(torch.cat([samples, time], 1))/60
print('Среднее новых точек:', samples.mean(0).tolist())
print('Разброс:', samples.std(0).tolist())
