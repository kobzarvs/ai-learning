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


# Создадим маленькую базу текстов, где записан лимит размера файла. По вопросу найдём подходящий фрагмент и вернём его вместе с указанием источника. Если нужных сведений нет, программа должна сообщить об этом. Это поисковая часть RAG (ответов с опорой на найденные документы); языковую модель, которая сочиняет ответ, здесь не подключаем.
documents = ['лимит загрузки файла 20 мегабайт', 'пароль можно изменить в настройках',
             'экспорт заметок доступен в меню холста']
vocab = {w:i for i,w in enumerate(sorted(set(' '.join(documents).split())))}
def vector(text):
    result = torch.zeros(len(vocab))
    for word in text.split():
        if word in vocab: result[vocab[word]] += 1
    return F.normalize(result, dim=0)
query = 'какой лимит загрузки файла'
scores = torch.stack([vector(d) for d in documents])@vector(query)
best = int(scores.argmax())
if scores[best] < 0.2:
    print('В базе нет достаточного основания для ответа.')
else:
    print('Найденный фрагмент:', documents[best])
    print('Источник: документ', best+1)
