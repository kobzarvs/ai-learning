# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
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

def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)

        # Сравниваем оценки классов с правильными метками; меньше — лучше.
        loss = F.cross_entropy(prediction, y)

        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()

        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()
    model.eval()

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    return float(loss.detach())

@torch.no_grad()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
def accuracy(model, x, y):
    scores = model(x)

    # argmax выбирает номер категории с наибольшей оценкой.
    predicted_labels = scores.argmax(dim=-1)

    # Для каждого объекта отмечаем, совпал ли прогноз с правильной меткой.
    correct = predicted_labels == y
    correct_numbers = correct.float()

    # Среднее нулей и единиц равно доле правильных ответов.
    correct_fraction = correct_numbers.mean()
    return correct_fraction.item()

# Создаём равномерную сетку между двумя границами, включая обе границы.
t = torch.linspace(0, 1, 64)

# Генерируем случайные целые числа; верхняя граница диапазона не включается.
y = torch.randint(0, 2, (300,))
x = torch.sin(2 * math.pi * (3 + 6*y[:, None]) * t)

# Создаём случайный шум из нормального распределения той же формы, что исходный тензор.
x = (x + 0.15 * torch.randn_like(x)).unsqueeze(1)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Свёртка ищет локальные закономерности вдоль последовательности.
model = nn.Sequential(nn.Conv1d(1, 12, 5, dilation=2), nn.SiLU(),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, 2))
fit_classifier(model, x[:240], y[:240])
print('Доля верных ответов:', accuracy(model, x[240:], y[240:]))
