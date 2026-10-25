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

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(100, 2)
y = x.sum(1, keepdim=True)
def run(seed):
    # Фиксируем начальное состояние генератора, чтобы повторить случайные данные.
    torch.manual_seed(seed)

    # Линейный слой учится преобразованию y = x @ W.T + b.
    model = nn.Linear(2, 1)
    fit_regression(model, x, y, steps=20)
    return model(x).detach()
a, b, c = run(7), run(7), run(8)

# Проверяем точное равенство формы и значений двух тензоров.
print('Одинаковый seed:', torch.equal(a, b))
print('Разница при другом seed:', (a-c).abs().max().item())
print('Версия PyTorch:', torch.__version__)
assert torch.equal(a, b)
