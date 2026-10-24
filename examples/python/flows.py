# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn

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
data = torch.randn(300)*0.5+3
# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
# Создаём тензор — числовой массив — из указанных значений.
shift = nn.Parameter(torch.tensor(0.))
log_scale = nn.Parameter(torch.tensor(0.))
# Adam подбирает величину шага для каждого параметра по истории градиентов.
optimizer = torch.optim.Adam([shift, log_scale], lr=0.05)
for _ in range(300):
    z = (data-shift)*torch.exp(-log_scale)
    # Минус логарифм плотности с поправкой на изменение масштаба.
    loss = (0.5*z.square()+log_scale).mean()
    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)
    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()
    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()
print('Сдвиг:', shift.item(), 'масштаб:', log_scale.exp().item())
z = torch.randn(5)
samples = shift + log_scale.exp()*z
recovered = (samples-shift)/log_scale.exp()
# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
print('Обратимость:', torch.allclose(z, recovered, atol=1e-5))
