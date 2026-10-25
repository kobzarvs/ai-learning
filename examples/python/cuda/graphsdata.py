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

# Создаём тензор — числовой массив — из указанных значений.
adjacency = torch.tensor([[1.,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]])
features = torch.tensor([[1.,0],[0,1],[1,1],[8,8]])
weights = adjacency / adjacency.sum(1, keepdim=True)

# Линейный слой учится преобразованию y = x @ W.T + b.
layer = nn.Linear(2, 3, bias=False)
messages = weights @ features
answer = layer(messages)
print('Сведения после обмена:', messages.tolist())
print('Изолированный объект:', messages[-1].tolist())

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
assert torch.allclose(messages[-1], features[-1])
