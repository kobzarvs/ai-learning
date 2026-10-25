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

# Создаём равномерную сетку между двумя границами, включая обе границы.
x = torch.linspace(-1, 1, 80)[:, None]
y = x.sin()

# Создаём тензор — числовой массив — из указанных значений.
query = torch.tensor([[-3.], [0.], [3.]])
predictions = []
for seed in range(5):
    # Фиксируем начальное состояние генератора, чтобы повторить случайные данные.
    torch.manual_seed(seed)

    # Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
    # Линейный слой учится преобразованию y = x @ W.T + b.
    model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    fit_regression(model, x, y)

    # Не записываем эти операции в граф для автоматического вычисления градиентов.
    with torch.no_grad(): predictions.append(model(query))

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
answers = torch.stack(predictions)
print('Средние ответы:', answers.mean(0).squeeze().tolist())
print('Разброс между моделями:', answers.std(0).squeeze().tolist())
