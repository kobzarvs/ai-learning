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

def points(count=320):
    # Создаём искусственные данные из нормального распределения со средним 0.
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

x, y = points(400)

# Линейный слой учится преобразованию y = x @ W.T + b.
global_model = nn.Linear(2, 2)
clients = [(x[:150], y[:150]), (x[150:300], y[150:300])]
for _ in range(15):
    local_models = []
    for local_x, local_y in clients:
        # Создаём независимую копию модели, чтобы дальнейшее обучение не меняло
        # оригинал.
        local = copy.deepcopy(global_model)
        fit_classifier(local, local_x, local_y, steps=5)
        local_models.append(local)
    with torch.no_grad():
        for name, parameter in global_model.named_parameters():
            # Собираем тензоры одинаковой формы в один, добавляя новое измерение.
            parameter.copy_(torch.stack([dict(m.named_parameters())[name] for m in local_models]).mean(0))
print('Качество общей модели:', accuracy(global_model, x[300:], y[300:]))
