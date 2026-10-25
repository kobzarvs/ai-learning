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

# Создаём искусственные данные из нормального распределения со средним 0.
population = torch.randn(100, 2)*3

# Создаём тензор — числовой массив — из указанных значений.
target = torch.tensor([1.5, -0.7])
for _ in range(50):
    score = (population-target).square().sum(1)

    # Находим k крайних значений и их индексы; largest=False выбирает наименьшие.
    elite = population[score.topk(10, largest=False).indices]
    children = elite.repeat(9, 1)+0.08*torch.randn(90, 2)

    # Соединяем тензоры вдоль уже существующего измерения dim.
    population = torch.cat([elite, children])
best = population[((population-target).square().sum(1)).argmin()]
print('Найденные параметры:', best.tolist())
