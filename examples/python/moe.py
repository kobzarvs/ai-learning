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
x = torch.randn(10, 4)
# Линейный слой учится преобразованию y = x @ W.T + b.
router = nn.Linear(4, 3)
# Регистрируем список подмоделей, чтобы учитывались параметры каждой из них.
experts = nn.ModuleList([nn.Linear(4, 2) for _ in range(3)])
# Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
scores = router(x).softmax(dim=1)
choice = scores.argmax(dim=1)
# Создаём тензор заданной формы и заполняем его нулями.
answer = torch.zeros(10, 2)
for i, expert in enumerate(experts):
    selected = choice == i
    answer[selected] = expert(x[selected])
print('Эксперт для каждого объекта:', choice.tolist())
# Считаем, сколько раз встретился каждый целочисленный индекс.
print('Объектов у экспертов:', torch.bincount(choice, minlength=3).tolist())
print('Ответов:', tuple(answer.shape))
