# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import copy
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

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(400, 2)
y_a = (x[:,0]>0).long()
y_b = (x[:,1]>0).long()

# Третий признак сообщает, какая именно задача запрошена.
a = torch.cat([x, torch.zeros(400, 1)], 1)

# Создаём тензор заданной формы и заполняем его единицами.
b = torch.cat([x, torch.ones(400, 1)], 1)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
first = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, 2))
fit_classifier(first, a[:300], y_a[:300])

# Создаём независимую копию модели, чтобы дальнейшее обучение не меняло оригинал.
naive, replay = copy.deepcopy(first), copy.deepcopy(first)
fit_classifier(naive, b[:300], y_b[:300])
fit_classifier(replay, torch.cat([b[:300], a[:80]]), torch.cat([y_b[:300], y_a[:80]]))
print('Старая задача после обычного дообучения:', accuracy(naive, a[300:], y_a[300:]))
print('Старая задача с повторением:', accuracy(replay, a[300:], y_a[300:]))
