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

def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)

        # Сравниваем оценки классов с правильными метками; меньше — лучше.
        loss = F.cross_entropy(prediction, y)

        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()

        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()
    model.eval()

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    return float(loss.detach())

texts = ['включи свет', 'свет включить', 'зажги лампу', 'лампу включить',
         'выключи свет', 'свет выключить', 'погаси лампу', 'лампу выключить']

# Создаём тензор — числовой массив — из указанных значений.
labels = torch.tensor([1,1,1,1,0,0,0,0])
vocab = {word:i for i,word in enumerate(sorted(set(' '.join(texts).split())))}
def encode(sentences):
    # Создаём тензор заданной формы и заполняем его нулями.
    result = torch.zeros(len(sentences), len(vocab))
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            if word in vocab:
                result[i, vocab[word]] += 1
    return result

# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Linear(len(vocab), 2)
fit_classifier(model, encode(texts), labels)
checks = ['включи лампу', 'выключи лампу', 'пожалуйста свет включить']

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    prediction = model(encode(checks)).argmax(1)
for sentence, answer in zip(checks, prediction):
    print(sentence, '→', 'включить' if answer else 'выключить')
