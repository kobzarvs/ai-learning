# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обнулить малые веса и проверить, что разреженность и уменьшение файла — разные вещи.
model = nn.Linear(20, 10)
x = torch.randn(30, 20)
with torch.no_grad():
    before = model(x)
    threshold = torch.quantile(model.weight.abs(), 0.5)
    model.weight.mul_(model.weight.abs()>=threshold)
    after = model(x)
print('Нулевых весов:', int((model.weight==0).sum()))
print('Плотная память весов в байтах:', model.weight.numel()*model.weight.element_size())
print('Ошибка после удаления:', F.mse_loss(after, before).item())
