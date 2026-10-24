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

# Соединяем тензоры вдоль уже существующего измерения dim.
# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.cat([torch.randn(100, 2)*0.3-1, torch.randn(100, 2)*0.3+1])
# Создаём случайную перестановку индексов для перемешивания объектов.
centers = x[torch.randperm(len(x))[:2]].clone()
for _ in range(20):
    # Вычисляем расстояния между каждой парой объектов из двух наборов.
    labels = torch.cdist(x, centers).argmin(1)
    for k in range(2):
        if (labels == k).any():
            centers[k] = x[labels == k].mean(0)
print('Центры групп:', centers.tolist())
# Считаем, сколько раз встретился каждый целочисленный индекс.
print('Число точек в группах:', torch.bincount(labels, minlength=2).tolist())
