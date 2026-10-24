# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

SUPPORTED_DEVICES = ['cpu', 'cuda', 'mps']
DEVICE_NOTE = 'GPU требует поддержки операций этого опыта установленной версией фреймворка. Автоматический переход на CPU при ошибке отключён.'

import argparse

parser = argparse.ArgumentParser(description="Учебный пример AI Atlas / PyTorch")
parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu",
                    help="cpu; NVIDIA cuda; Apple Metal mps; auto выбирает доступный GPU")
requested_device = parser.parse_args().device
available = {"cpu": True, "cuda": torch.cuda.is_available(),
             "mps": torch.backends.mps.is_available()}
if requested_device == "auto":
    requested_device = next(d for d in ("cuda", "mps", "cpu")
                            if d in SUPPORTED_DEVICES and available[d])
if requested_device not in SUPPORTED_DEVICES:
    parser.error("Для этого опыта доступны: " + ", ".join(SUPPORTED_DEVICES)
                 + ". " + DEVICE_NOTE)
if not available[requested_device]:
    parser.error("Устройство " + requested_device + " недоступно. "
                 "CUDA требует NVIDIA, драйвер и CUDA-сборку PyTorch; "
                 "MPS требует совместимый Mac и нативный запуск вне Linux-контейнера. "
                 "Для запуска на процессоре явно укажи --device cpu.")
device = torch.device(requested_device)
# Новые данные И параметры моделей создаются на одном выбранном устройстве.
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device, "| dtype: float32")

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
