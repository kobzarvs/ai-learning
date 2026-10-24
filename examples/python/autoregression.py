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

# Обучить сеть предсказывать следующий символ в короткой циклической последовательности.
sequence = 'abcabcabcabcabcabc'
vocab = {c:i for i,c in enumerate('abc')}
tokens = torch.tensor([vocab[c] for c in sequence])
x, y = tokens[:-1], tokens[1:]
model = nn.Sequential(nn.Embedding(3, 8), nn.Linear(8, 3))
fit_classifier(model, x, y)
current, generated = torch.tensor([0]), 'a'
with torch.no_grad():
    for _ in range(12):
        current = model(current).argmax(1)
        generated += 'abc'[int(current)]
print('Продолжение:', generated)
