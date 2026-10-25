# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна: проверь видеокарту NVIDIA и установку PyTorch.")

device = torch.device("cuda")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

# До опыта допускаем все вероятности равномерно: Beta(1, 1).
a, b = 1., 1.

# Создаём тензор — числовой массив — из указанных значений.
observations = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])
a += observations.sum().item()
b += (1-observations).sum().item()

# Бета-распределение описывает неопределённость вероятности от 0 до 1.
posterior = torch.distributions.Beta(a, b)
samples = posterior.sample((10000,))

# Находим пороги, ниже которых лежат заданные доли значений.
interval = torch.quantile(samples, torch.tensor([0.025, 0.975]))
print('Оценка вероятности:', a/(a+b))
print('95% интервал:', interval.tolist())
