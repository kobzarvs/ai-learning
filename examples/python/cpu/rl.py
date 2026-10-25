# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
model = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 2))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

# Собираем все переходы маленькой среды. Действие 0 — влево, 1 — вправо.
states, actions, next_states, rewards, done = [], [], [], [], []
for state in range(4):
    for action in range(2):
        nxt = max(0, min(4, state+(1 if action else -1)))
        states.append(state)
        actions.append(action)
        next_states.append(nxt)
        rewards.append(1.0 if nxt==4 else -0.02)
        done.append(nxt==4)

# Превращаем номер категории в вектор: единица на её месте, остальные — нули.
# Создаём тензор — числовой массив — из указанных значений.
s = F.one_hot(torch.tensor(states), 5).float()
ns = F.one_hot(torch.tensor(next_states), 5).float()
a = torch.tensor(actions)
reward = torch.tensor(rewards)
terminal = torch.tensor(done)
for _ in range(300):
    # Не записываем эти операции в граф для автоматического вычисления градиентов.
    with torch.no_grad():
        target = reward+0.95*model(ns).max(1).values*(~terminal)
    value = model(s).gather(1, a[:, None]).squeeze(1)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(value, target)

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()
with torch.no_grad():
    # Создаём единичную матрицу: единицы на диагонали, остальные элементы — нули.
    print('Действия в клетках 0–3:', model(torch.eye(5)[:4]).argmax(1).tolist())
