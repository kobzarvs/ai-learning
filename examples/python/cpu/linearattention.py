# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

# Генерируем случайные числа, равномерно распределённые от 0 до 1.
q, k, v = [torch.rand(12, 8) for _ in range(3)]

# Положительные признаки делают знаменатель устойчивым.
phi = lambda x: F.elu(x) + 1
q, k = phi(q), phi(k)

# Создаём тензор заданной формы и заполняем его нулями.
memory, normalizer = torch.zeros(8, 8), torch.zeros(8)
answers = []
for t in range(len(q)):
    # Внешнее произведение: получаем матрицу всех попарных произведений двух векторов.
    memory += torch.outer(k[t], v[t])
    normalizer += k[t]
    answers.append((q[t] @ memory) / (q[t] @ normalizer + 1e-6))

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
recurrent = torch.stack(answers)
scores = (q @ k.T).tril()
reference = (scores @ v) / (scores.sum(1, keepdim=True) + 1e-6)
print('Максимальная разница:', (recurrent-reference).abs().max().item())

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
assert torch.allclose(recurrent, reference, atol=1e-5)
