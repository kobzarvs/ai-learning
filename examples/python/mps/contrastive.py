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
x = torch.randn(128, 8)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
encoder = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 12))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.01)
for _ in range(100):
    # Нормируем векторы; по умолчанию их евклидова длина становится равной 1.
    # Создаём случайный шум из нормального распределения той же формы, что исходный
    # тензор.
    a = F.normalize(encoder(x + 0.1*torch.randn_like(x)), dim=1)
    b = F.normalize(encoder(x + 0.1*torch.randn_like(x)), dim=1)
    scores = a @ b.T / 0.1

    # Сравниваем оценки классов с правильными метками; меньше — лучше.
    # Создаём последовательность чисел с постоянным шагом; конечная граница не
    # включается.
    loss = F.cross_entropy(scores, torch.arange(len(x)))

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    new = torch.randn(40, 8)
    a = F.normalize(encoder(new), dim=1)
    b = F.normalize(encoder(new + 0.1*torch.randn_like(new)), dim=1)
    print('Найден свой второй вид:', (a @ b.T).argmax(1).eq(torch.arange(40)).float().mean().item())
