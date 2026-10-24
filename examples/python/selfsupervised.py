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

# Создаём искусственные данные из нормального распределения со средним 0.
t = torch.randn(320, 1)
# Соединяем тензоры вдоль уже существующего измерения dim.
x = torch.cat([t, 2*t, -t, 0.5*t], 1)+0.03*torch.randn(320, 4)
# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 4))
# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(220):
    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    mask = torch.rand(250, 4)<0.35
    # Заменяем элементы, отмеченные логической маской, указанным значением.
    observed = x[:250].masked_fill(mask, 0)
    prediction = model(torch.cat([observed, mask.float()], 1))
    # Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).
    loss = ((prediction-x[:250]).square()*mask).sum()/mask.sum().clamp_min(1)
    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)
    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()
    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()
# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    # Создаём тензор заданной формы и заполняем его нулями.
    mask = torch.zeros(70, 4, dtype=torch.bool)
    mask[:, 1] = True
    prediction = model(torch.cat([x[250:].masked_fill(mask, 0), mask.float()], 1))
# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка на скрытом признаке новых объектов:', F.mse_loss(prediction[:,1], x[250:,1]).item())
