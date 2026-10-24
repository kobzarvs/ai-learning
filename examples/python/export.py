# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Экспортировать вычисление через актуальный torch.export и проверить совпадение результата.
model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2)).eval()
x = torch.randn(3, 4)
program = torch.export.export(model, (x,))
with torch.no_grad():
    original = model(x)
    exported = program.module()(x)
print('Совпадение:', torch.allclose(original, exported))
print('Узлов графа:', len(list(program.graph.nodes)))
assert torch.allclose(original, exported)
# Для файла можно вызвать torch.export.save(program, 'model.pt2').
