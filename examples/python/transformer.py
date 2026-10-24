# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
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

# 2 последовательности, 6 позиций, 16 признаков.
x = torch.randn(2, 6, 16)

# Линейный слой учится преобразованию y = x @ W.T + b.
projection = nn.Linear(16, 48, bias=False)
q, k, v = projection(x).chunk(3, dim=-1)

# Измерения: набор, голова, позиция, признаки головы.
q, k, v = [z.view(2, 6, 4, 4).transpose(1, 2) for z in (q, k, v)]

# Внимание взвешивает значения V по сходству запросов Q и ключей K.
answer = F.scaled_dot_product_attention(q, k, v, is_causal=True)
changed_v = v.clone()
changed_v[:, :, -1] += 100
changed = F.scaled_dot_product_attention(q, k, changed_v, is_causal=True)
print('Размер ответа:', tuple(answer.shape))

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
print('Ранние ответы не меняются:', torch.allclose(answer[:, :, :-1], changed[:, :, :-1]))
assert torch.allclose(answer[:, :, :-1], changed[:, :, :-1])
