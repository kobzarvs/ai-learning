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

# Создаём тензор заданной формы и заполняем его нулями.
x = torch.zeros(30)
x[3] = 1
state = torch.zeros(4)

# Создаём тензор — числовой массив — из указанных значений.
decay = torch.tensor([0.1, 0.5, 0.8, 0.95])
readout = torch.tensor([0.25, 0.25, 0.25, 0.25])
outputs = []
for value in x:
    state = decay * state + value
    outputs.append((readout * state).sum().item())
print('Ответ после импульса:', [round(v, 3) for v in outputs[3:12]])
assert outputs[3] > outputs[8] > outputs[20] > 0
