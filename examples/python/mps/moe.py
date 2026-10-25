# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS недоступен: нужен совместимый Mac и нативный запуск.")

device = torch.device("mps")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(10, 4)

# Линейный слой учится преобразованию y = x @ W.T + b.
router = nn.Linear(4, 3)

# Регистрируем список подмоделей, чтобы учитывались параметры каждой из них.
experts = nn.ModuleList([nn.Linear(4, 2) for _ in range(3)])

# Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
scores = router(x).softmax(dim=1)
choice = scores.argmax(dim=1)

# Создаём тензор заданной формы и заполняем его нулями.
answer = torch.zeros(10, 2)
for i, expert in enumerate(experts):
    selected = choice == i
    answer[selected] = expert(x[selected])
print('Эксперт для каждого объекта:', choice.tolist())

# Считаем, сколько раз встретился каждый целочисленный индекс.
print('Объектов у экспертов:', torch.bincount(choice, minlength=3).tolist())
print('Ответов:', tuple(answer.shape))
