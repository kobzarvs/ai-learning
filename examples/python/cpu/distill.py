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

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
teacher = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
fit_classifier(teacher, x[:300], y[:300])

# Явно включаем или отключаем вычисление градиентов для этих параметров.
teacher.eval().requires_grad_(False)
student = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.03)
temperature = 2.
for _ in range(180):
    with torch.no_grad():
        # Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
        soft_target = (teacher(x[:300])/temperature).softmax(1)

    # KL-дивергенция измеряет расхождение распределений ученика и учителя.
    # Устойчиво вычисляем логарифмы вероятностей классов из исходных оценок.
    soft_loss = F.kl_div((student(x[:300])/temperature).log_softmax(1), soft_target, reduction='batchmean')*temperature**2
    loss = 0.7*soft_loss+0.3*F.cross_entropy(student(x[:300]), y[:300])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print('Учитель:', accuracy(teacher, x[300:], y[300:]))
print('Ученик:', accuracy(student, x[300:], y[300:]))
