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

# В каждой задаче своё правило y=a*x+b. Контекст: две пары (x,y) и новый x.
def tasks(count):
    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    a, b = torch.rand(count, 1)*2-1, torch.rand(count, 1)*2-1

    # Создаём тензор заданной формы и заполняем его единицами.
    x1, x2 = -torch.ones(count, 1), torch.ones(count, 1)
    query = torch.rand(count, 1)*2-1

    # Соединяем тензоры вдоль уже существующего измерения dim.
    context = torch.cat([x1, a*x1+b, x2, a*x2+b, query], 1)
    return context, a*query+b

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 1))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(500):
    x, y = tasks(128)
    prediction = model(x)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(prediction, y)

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()
model.eval()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    x, y = tasks(100)
    prediction = model(x)
    test_error = F.mse_loss(prediction, y)
    print('Ошибка на новых правилах без дообучения:', test_error.item())
