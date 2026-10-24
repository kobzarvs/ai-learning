# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Оценить категорию по распределению каждого отдельного признака.
x = torch.cat([torch.randn(120, 2)-1, torch.randn(120, 2)+1])
y = torch.cat([torch.zeros(120), torch.ones(120)]).long()
order = torch.randperm(len(x)); x, y = x[order], y[order]
means, variances, priors = [], [], []
for label in range(2):
    subset = x[:180][y[:180] == label]
    means.append(subset.mean(0)); variances.append(subset.var(0)+1e-4)
    priors.append(len(subset)/180)
mu, var = torch.stack(means), torch.stack(variances)
log_score = -0.5*((x[180:, None]-mu).square()/var+var.log()).sum(2)
log_score += torch.tensor(priors).log()
print('Верно:', log_score.argmax(1).eq(y[180:]).float().mean().item())
