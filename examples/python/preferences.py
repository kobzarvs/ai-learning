# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить предпочтение одному ответу над другим через парную ошибку DPO.
logits = nn.Parameter(torch.zeros(2))
reference = torch.zeros(2).log_softmax(0).detach()
optimizer = torch.optim.Adam([logits], lr=0.05)
for _ in range(100):
    logp = logits.log_softmax(0)
    # В каждой паре ответ 1 предпочитается ответу 0.
    advantage = (logp[1]-logp[0])-(reference[1]-reference[0])
    loss = -F.logsigmoid(0.5*advantage)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Вероятности ответов:', logits.softmax(0).detach().tolist())
