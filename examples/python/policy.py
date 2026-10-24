# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить вероятностный выбор действия непосредственно по награде.
logits = nn.Parameter(torch.zeros(3))
optimizer = torch.optim.Adam([logits], lr=0.04)
reward_table = torch.tensor([0., 0.4, 1.])
for _ in range(250):
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample((64,))
    rewards = reward_table[actions]
    advantage = rewards-rewards.mean()
    loss = -(distribution.log_prob(actions)*advantage.detach()).mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Вероятности действий:', logits.softmax(0).detach().tolist())
