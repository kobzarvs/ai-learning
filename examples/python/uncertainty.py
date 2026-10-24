# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


def fit_regression(model, x, y, steps=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())

# Сравнить ответы нескольких моделей на знакомых и далёких от обучения объектах.
x = torch.linspace(-1, 1, 80)[:, None]; y = x.sin()
query = torch.tensor([[-3.], [0.], [3.]])
predictions = []
for seed in range(5):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    fit_regression(model, x, y)
    with torch.no_grad(): predictions.append(model(query))
answers = torch.stack(predictions)
print('Средние ответы:', answers.mean(0).squeeze().tolist())
print('Разброс между моделями:', answers.std(0).squeeze().tolist())
