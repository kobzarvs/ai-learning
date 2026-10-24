# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Получить прогноз функции и разброс там, где наблюдений мало.
x = torch.tensor([[-1.], [0.], [1.]])
y = x.sin()
query = torch.linspace(-3, 3, 21)[:, None]
def kernel(a, b):
    return torch.exp(-0.5*torch.cdist(a, b).square())
k = kernel(x, x)+0.01*torch.eye(len(x))
cross = kernel(query, x)
mean = cross @ torch.linalg.solve(k, y)
variance = (1-(cross*torch.linalg.solve(k, cross.T).T).sum(1)).clamp_min(0)
for i in [0, 10, 20]:
    print('x =', query[i].item(), 'среднее =', mean[i].item(), 'разброс =', variance[i].sqrt().item())
