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

sequence = 'abcabcabcabcabcabc'
vocab = {c:i for i,c in enumerate('abc')}

# Создаём тензор — числовой массив — из указанных значений.
tokens = torch.tensor([vocab[c] for c in sequence])
x, y = tokens[:-1], tokens[1:]

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Каждому целочисленному индексу сопоставляем обучаемый вектор.
model = nn.Sequential(nn.Embedding(3, 8), nn.Linear(8, 3))
fit_classifier(model, x, y)
current, generated = torch.tensor([0]), 'a'

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    for _ in range(12):
        current = model(current).argmax(1)
        generated += 'abc'[int(current)]
print('Продолжение:', generated)
