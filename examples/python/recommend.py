# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# По известным оценкам научить маленькие векторы пользователей и предметов предсказывать совместимость.
true_users, true_items = torch.randn(12, 3), torch.randn(10, 3)
rating = true_users @ true_items.T
users, items = nn.Embedding(12, 3), nn.Embedding(10, 3)
mask = torch.rand(12, 10) < 0.8
optimizer = torch.optim.AdamW(list(users.parameters())+list(items.parameters()), lr=0.04)
for _ in range(400):
    predictions = users.weight @ items.weight.T
    loss = F.mse_loss(predictions[mask], rating[mask])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = users.weight @ items.weight.T
print('Ошибка на скрытых оценках:', F.mse_loss(prediction[~mask], rating[~mask]).item())
