# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import copy
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

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(400, 2)
y_a = (x[:,0]>0).long()
y_b = (x[:,1]>0).long()
# Третий признак сообщает, какая именно задача запрошена.
a = torch.cat([x, torch.zeros(400, 1)], 1)
# Создаём тензор заданной формы и заполняем его единицами.
b = torch.cat([x, torch.ones(400, 1)], 1)
# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
first = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, 2))
fit_classifier(first, a[:300], y_a[:300])
# Создаём независимую копию модели, чтобы дальнейшее обучение не меняло оригинал.
naive, replay = copy.deepcopy(first), copy.deepcopy(first)
fit_classifier(naive, b[:300], y_b[:300])
fit_classifier(replay, torch.cat([b[:300], a[:80]]), torch.cat([y_b[:300], y_a[:80]]))
print('Старая задача после обычного дообучения:', accuracy(naive, a[300:], y_a[300:]))
print('Старая задача с повторением:', accuracy(replay, a[300:], y_a[300:]))
