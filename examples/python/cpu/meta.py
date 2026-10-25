# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import copy
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

# Линейный слой учится преобразованию y = x @ W.T + b.
initial = nn.Linear(1, 1)
for _ in range(120):
    # Создаём независимую копию модели, чтобы дальнейшее обучение не меняло оригинал.
    task = copy.deepcopy(initial)

    # Создаём искусственные данные из нормального распределения со средним 0.
    x = torch.randn(12, 1)

    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    slope = 1+torch.rand(())
    bias = torch.rand(())*0.4
    fit_regression(task, x, slope*x+bias, steps=10)

    # Не записываем эти операции в граф для автоматического вычисления градиентов.
    with torch.no_grad():
        for shared, adapted in zip(initial.parameters(), task.parameters()):
            shared.lerp_(adapted, 0.15)
x = torch.randn(30, 1)
y = 1.7*x+0.15
adapted = copy.deepcopy(initial)
fit_regression(adapted, x[:5], y[:5], steps=10)
with torch.no_grad():
    print('До адаптации:', F.mse_loss(initial(x[5:]), y[5:]).item())
    print('После 10 шагов:', F.mse_loss(adapted(x[5:]), y[5:]).item())
