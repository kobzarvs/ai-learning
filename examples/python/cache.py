# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
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


# Будем обрабатывать последовательность по одному элементу. Для механизма внимания сохраним уже рассчитанные ключи (описания, по которым находят нужные предыдущие элементы) и значения (сведения, которые они передают). Такое временное хранилище называется кэшем. Проверим, что повторное использование сохранённых данных даёт те же ответы, что расчёт всей последовательности сразу.
x = torch.randn(1, 1, 10, 8)
q, k, v = x+0.1, x-0.1, x*0.5
full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
k_cache, v_cache, outputs = [], [], []
for t in range(10):
    k_cache.append(k[:, :, t:t+1]); v_cache.append(v[:, :, t:t+1])
    # У запроса длины 1 все сохранённые позиции уже в прошлом.
    output = F.scaled_dot_product_attention(q[:, :, t:t+1],
        torch.cat(k_cache, dim=2), torch.cat(v_cache, dim=2), is_causal=False)
    outputs.append(output)
cached = torch.cat(outputs, dim=2)
print('Совпадает с полным расчётом:', torch.allclose(full, cached, atol=1e-6))
assert torch.allclose(full, cached, atol=1e-6)
