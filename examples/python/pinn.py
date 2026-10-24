# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_DEVICES = ['cpu', 'cuda', 'mps']
DEVICE_NOTE = 'GPU требует поддержки операций этого опыта установленной версией фреймворка. Автоматический переход на CPU при ошибке отключён.'

import argparse

parser = argparse.ArgumentParser(description="Учебный пример AI Atlas / PyTorch")
parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu",
                    help="cpu; NVIDIA cuda; Apple Metal mps; auto выбирает доступный GPU")
requested_device = parser.parse_args().device
available = {"cpu": True, "cuda": torch.cuda.is_available(),
             "mps": torch.backends.mps.is_available()}
if requested_device == "auto":
    requested_device = next(d for d in ("cuda", "mps", "cpu")
                            if d in SUPPORTED_DEVICES and available[d])
if requested_device not in SUPPORTED_DEVICES:
    parser.error("Для этого опыта доступны: " + ", ".join(SUPPORTED_DEVICES)
                 + ". " + DEVICE_NOTE)
if not available[requested_device]:
    parser.error("Устройство " + requested_device + " недоступно. "
                 "CUDA требует NVIDIA, драйвер и CUDA-сборку PyTorch; "
                 "MPS требует совместимый Mac и нативный запуск вне Linux-контейнера. "
                 "Для запуска на процессоре явно укажи --device cpu.")
device = torch.device(requested_device)
# Новые данные И параметры моделей создаются на одном выбранном устройстве.
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device, "| dtype: float32")

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
    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    print('Ошибка относительно exp(−x):', F.mse_loss(model(x), torch.exp(-x)).item())
