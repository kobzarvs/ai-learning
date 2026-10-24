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


def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

# Возьмём короткие команды двух видов и превратим слова в числа, понятные программе. Научим маленькую модель определять нужный вид команды. Затем проверим новые формулировки, которые не входили в учебный набор. Это распознавание категории текста, а не собеседник, который понимает любой вопрос.
texts = ['включи свет', 'свет включить', 'зажги лампу', 'лампу включить',
         'выключи свет', 'свет выключить', 'погаси лампу', 'лампу выключить']
labels = torch.tensor([1,1,1,1,0,0,0,0])
vocab = {word:i for i,word in enumerate(sorted(set(' '.join(texts).split())))}
def encode(sentences):
    result = torch.zeros(len(sentences), len(vocab))
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            if word in vocab:
                result[i, vocab[word]] += 1
    return result
model = nn.Linear(len(vocab), 2)
fit_classifier(model, encode(texts), labels)
checks = ['включи лампу', 'выключи лампу', 'пожалуйста свет включить']
with torch.no_grad():
    prediction = model(encode(checks)).argmax(1)
for sentence, answer in zip(checks, prediction):
    print(sentence, '→', 'включить' if answer else 'выключить')
