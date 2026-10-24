# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Обучить маленькую сеть выбирать шаг в коридоре из пяти клеток, чтобы добраться до награды.
model = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
# Собираем все переходы маленькой среды. Действие 0 — влево, 1 — вправо.
states, actions, next_states, rewards, done = [], [], [], [], []
for state in range(4):
    for action in range(2):
        nxt = max(0, min(4, state+(1 if action else -1)))
        states.append(state); actions.append(action); next_states.append(nxt)
        rewards.append(1.0 if nxt==4 else -0.02); done.append(nxt==4)
s = F.one_hot(torch.tensor(states), 5).float()
ns = F.one_hot(torch.tensor(next_states), 5).float()
a = torch.tensor(actions); reward = torch.tensor(rewards); terminal = torch.tensor(done)
for _ in range(300):
    with torch.no_grad():
        target = reward+0.95*model(ns).max(1).values*(~terminal)
    value = model(s).gather(1, a[:, None]).squeeze(1)
    loss = F.mse_loss(value, target)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    print('Действия в клетках 0–3:', model(torch.eye(5)[:4]).argmax(1).tolist())
