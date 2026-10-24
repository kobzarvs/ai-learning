# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Выбирать одно из трёх действий, постепенно выясняя, какое чаще приносит награду.
true_chance = torch.tensor([0.2, 0.5, 0.8])
success = torch.ones(3); failure = torch.ones(3)
count = torch.zeros(3, dtype=torch.long)
for _ in range(800):
    guesses = torch.distributions.Beta(success, failure).sample()
    action = int(guesses.argmax())
    reward = int(torch.rand(()) < true_chance[action])
    success[action] += reward; failure[action] += 1-reward
    count[action] += 1
print('Сколько раз выбрано каждое действие:', count.tolist())
print('Оценки вероятностей:', (success/(success+failure)).tolist())
