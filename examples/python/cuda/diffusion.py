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

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(250):
    # Создаём искусственные данные из нормального распределения со средним 0.
    clean = torch.randn(128, 1)*0.3+2

    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    sigma = torch.rand(128, 1)*0.8+0.05

    # Создаём случайный шум из нормального распределения той же формы, что исходный
    # тензор.
    noisy = clean + sigma*torch.randn_like(clean)

    # В каждой строке входа — зашумлённое число и сила добавленного шума.
    model_input = torch.cat([noisy, sigma], dim=1)
    prediction = model(model_input)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(prediction, clean)

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    clean = torch.randn(100, 1)*0.3+2

    # Создаём тензор той же формы и заполняем его одной заданной величиной.
    sigma = torch.full_like(clean, 0.5)
    noisy = clean+sigma*torch.randn_like(clean)
    restored = model(torch.cat([noisy, sigma], 1))
print('До очистки:', F.mse_loss(noisy, clean).item())
print('После очистки:', F.mse_loss(restored, clean).item())
