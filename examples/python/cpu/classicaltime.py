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

def fit_regression(model, x, y, steps=200):
    # AdamW обновляет параметры по градиентам.
    # Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)

        # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
        loss = F.mse_loss(prediction, y)

        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()

        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()
    model.eval()

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    return float(loss.detach())

# Создаём последовательность чисел с постоянным шагом; конечная граница не включается.
# Создаём искусственные данные из нормального распределения со средним 0.
signal = torch.sin(torch.arange(300)*0.1)+0.05*torch.randn(300)

# Нарезаем последовательность на скользящие окна заданной длины.
windows = signal.unfold(0, 7, 1)
x, y = windows[:, :6], windows[:, 6:]

# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Linear(6, 1)
fit_regression(model, x[:220], y[:220])

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    prediction = model(x[220:])
print('Ошибка модели:', F.mse_loss(prediction, y[220:]).item())
print('Ошибка «повтори последнее»:', F.mse_loss(x[220:, -1:], y[220:]).item())
