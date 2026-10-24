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

def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)
        # Сравниваем оценки классов с правильными метками; меньше — лучше.
        loss = F.cross_entropy(model(x), y)
        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()
        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()
    model.eval()
    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    return float(loss.detach())

@torch.no_grad()
# Не записываем эти операции в граф для автоматического вычисления градиентов.
def accuracy(model, x, y):
    # argmax выбирает номер категории с наибольшей оценкой.
    return float((model(x).argmax(dim=-1) == y).float().mean())

def points(count=320):
    # Создаём искусственные данные из нормального распределения со средним 0.
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y

x, y = points(400)
# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
teacher = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
fit_classifier(teacher, x[:300], y[:300])
# Явно включаем или отключаем вычисление градиентов для этих параметров.
teacher.eval().requires_grad_(False)
student = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.03)
temperature = 2.
for _ in range(180):
    with torch.no_grad():
        # Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
        soft_target = (teacher(x[:300])/temperature).softmax(1)
    # KL-дивергенция измеряет расхождение распределений ученика и учителя.
    # Устойчиво вычисляем логарифмы вероятностей классов из исходных оценок.
    soft_loss = F.kl_div((student(x[:300])/temperature).log_softmax(1), soft_target, reduction='batchmean')*temperature**2
    loss = 0.7*soft_loss+0.3*F.cross_entropy(student(x[:300]), y[:300])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print('Учитель:', accuracy(teacher, x[300:], y[300:]))
print('Ученик:', accuracy(student, x[300:], y[300:]))
