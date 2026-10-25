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

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2)).eval()

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(3, 4)

# Записываем вычислительный граф модели для последующего выполнения.
program = torch.export.export(model, (x,))

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    original = model(x)
    exported = program.module()(x)

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
print('Совпадение:', torch.allclose(original, exported))
print('Узлов графа:', len(list(program.graph.nodes)))
assert torch.allclose(original, exported)

# Для файла можно вызвать torch.export.save(program, 'model.pt2').
