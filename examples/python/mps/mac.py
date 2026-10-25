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

# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Linear(2, 1).to(device)

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(64, 2, device=device)
y = x.sum(1, keepdim=True)

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(150):
    prediction = model(x)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(prediction, y)

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# item() дожидается готовности результата и возвращает обычное число.
print('Устройство:', device, 'ошибка:', loss.detach().item())
