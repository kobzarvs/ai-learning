# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


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

# Обучить компактный классификатор двух команд по словам и проверить новые формулировки.
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
