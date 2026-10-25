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
x = torch.cat([torch.randn(120)*0.3-1.5, torch.randn(120)*0.3+1.5])

# Создаём тензор — числовой массив — из указанных значений.
means = torch.tensor([-1., 1.])

# Создаём тензор заданной формы и заполняем его единицами.
variance = torch.ones(2)
weights = torch.ones(2)/2
for _ in range(30):
    score = -0.5*((x[:, None]-means).square()/variance+variance.log())+weights.log()

    # Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
    responsibility = score.softmax(1)
    count = responsibility.sum(0)
    means = (responsibility*x[:, None]).sum(0)/count
    variance = (responsibility*(x[:, None]-means).square()).sum(0)/count

    # Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).
    variance = variance.clamp_min(1e-4)
    weights = count/len(x)
print('Центры:', means.tolist(), 'доли:', weights.tolist())
