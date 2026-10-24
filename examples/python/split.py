# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Отделить подготовку данных от проверки, чтобы будущие значения не влияли на обучение.
x = torch.randn(300, 3)
x[240:] += 1.5  # Более поздние данные пришли из других условий.
train, validation, test = x[:180], x[180:240], x[240:]
mean = train.mean(0); std = train.std(0).clamp_min(1e-6)
train_scaled = (train-mean)/std
test_scaled = (test-mean)/std
print('Среднее учебной части:', train_scaled.mean(0).tolist())
print('Среднее теста с учебными настройками:', test_scaled.mean(0).tolist())
assert train_scaled.mean(0).abs().max()<1e-6
