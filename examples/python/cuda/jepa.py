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

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(256, 4)
context = x[:, :2]

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
target_part = torch.stack([x[:,0]+x[:,1], x[:,0]-x[:,1]], 1)

# Фиксированный целевой encoder не может схлопнуться в одну константу.
target_encoder = nn.Linear(2, 8, bias=False).requires_grad_(False)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# SiLU плавно ослабляет отрицательные значения и сохраняет большие положительные.
# Такая нелинейная функция позволяет сети учить сложные зависимости.
predictor = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 8))

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    target = target_encoder(target_part)
fit_regression(predictor, context[:200], target[:200])
with torch.no_grad():
    print('Ошибка предсказания представления:', F.mse_loss(predictor(context[200:]), target[200:]).item())
