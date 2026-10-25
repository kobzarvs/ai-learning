# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

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

class ContinuousCell(nn.Module):
    def __init__(self):
        super().__init__()

        # Линейный слой учится преобразованию y = x @ W.T + b.
        self.target = nn.Linear(2, 4)
        self.speed = nn.Linear(2, 4)
    def forward(self, state, x, dt):
        target = self.target(x).tanh()

        # Softplus плавно преобразует число в положительное значение.
        decay = torch.exp(-F.softplus(self.speed(x))*dt)
        return decay*state + (1-decay)*target
cell = ContinuousCell()

# Создаём тензор заданной формы и заполняем его единицами.
# Создаём тензор заданной формы и заполняем его нулями.
x, state = torch.ones(1, 2), torch.zeros(1, 4)
for dt in [0.01, 0.1, 1.0]:
    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    print('Интервал', dt, 'состояние', cell(state, x, dt).detach().tolist())
