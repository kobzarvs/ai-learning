# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

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
