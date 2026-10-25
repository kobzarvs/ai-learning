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

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 1))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(400):
    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    x = torch.rand(80, 1, requires_grad=True)*2
    u = model(x)

    # Получаем производную выхода по входу.
    # create_graph=True позволяет затем вычислить производную ещё раз.
    derivative, = torch.autograd.grad(u.sum(), x, create_graph=True)
    equation_error = (derivative+u).square().mean()

    # Создаём тензор заданной формы и заполняем его нулями.
    boundary_error = (model(torch.zeros(1,1))-1).square().mean()
    loss = equation_error+boundary_error

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    # Создаём равномерную сетку между двумя границами, включая обе границы.
    x = torch.linspace(0, 2, 50)[:, None]
    prediction = model(x)
    reference = torch.exp(-x)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    test_error = F.mse_loss(prediction, reference)
    print('Ошибка относительно exp(−x):', test_error.item())
