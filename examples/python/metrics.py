# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Увидеть, как 95% точности могут скрывать полный провал поиска редких событий.
y = torch.tensor([0]*95+[1]*5)
prediction = torch.zeros_like(y)
true_positive = ((prediction==1)&(y==1)).sum()
false_positive = ((prediction==1)&(y==0)).sum()
false_negative = ((prediction==0)&(y==1)).sum()
accuracy_value = prediction.eq(y).float().mean()
precision = true_positive/(true_positive+false_positive).clamp_min(1)
recall = true_positive/(true_positive+false_negative).clamp_min(1)
print('Общая точность:', accuracy_value.item())
print('Точность тревог:', precision.item(), 'доля найденных событий:', recall.item())
