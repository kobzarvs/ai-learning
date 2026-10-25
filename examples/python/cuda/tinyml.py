# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn

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
x = torch.randn(1, 16, 24, 24)

# Свёртка применяет обучаемые фильтры к небольшим участкам изображения.
regular = nn.Conv2d(16, 32, 3, padding=1, bias=False)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
compact = nn.Sequential(
    nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
    nn.Conv2d(16, 32, 1, bias=False),
)
for name, model in [('Обычная', regular), ('Компактная', compact)]:
    count = sum(p.numel() for p in model.parameters())
    print(name, 'параметров:', count, 'байт весов:', count*4,
          'размер ответа:', tuple(model(x).shape))
