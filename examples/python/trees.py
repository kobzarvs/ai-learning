# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

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

def points(count=320):
    # Создаём искусственные данные из нормального распределения со средним 0.
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

x, y = points(240)
best = None
for feature in range(2):
    # Создаём равномерную сетку между двумя границами, включая обе границы.
    for threshold in torch.linspace(-2, 2, 30):
        left = x[:180, feature] < threshold
        if not left.any() or left.all():
            continue
        a = y[:180][left].float().mean() >= 0.5
        b = y[:180][~left].float().mean() >= 0.5
        # Для каждого элемента выбираем одно из двух значений по логическому условию.
        predicted = torch.where(left, a, b)
        error = (predicted != y[:180]).float().mean().item()
        if best is None or error < best[0]:
            best = (error, feature, threshold, a, b)
_, feature, threshold, a, b = best
predicted = torch.where(x[180:, feature] < threshold, a, b)
print('Признак:', feature, 'порог:', threshold.item())
print('Верно на новых строках:', (predicted == y[180:]).float().mean().item())
