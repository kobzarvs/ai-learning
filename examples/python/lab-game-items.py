# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_DEVICES = ['cpu', 'cuda', 'mps']
DEVICE_NOTE = 'GPU требует поддержки операций этого опыта установленной версией фреймворка. Автоматический переход на CPU при ошибке отключён.'

import argparse

parser = argparse.ArgumentParser(description="Учебный пример AI Atlas / PyTorch")
parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu",
                    help="cpu; NVIDIA cuda; Apple Metal mps; auto выбирает доступный GPU")
requested_device = parser.parse_args().device
available = {"cpu": True, "cuda": torch.cuda.is_available(),
             "mps": torch.backends.mps.is_available()}
if requested_device == "auto":
    requested_device = next(d for d in ("cuda", "mps", "cpu")
                            if d in SUPPORTED_DEVICES and available[d])
if requested_device not in SUPPORTED_DEVICES:
    parser.error("Для этого опыта доступны: " + ", ".join(SUPPORTED_DEVICES)
                 + ". " + DEVICE_NOTE)
if not available[requested_device]:
    parser.error("Устройство " + requested_device + " недоступно. "
                 "CUDA требует NVIDIA, драйвер и CUDA-сборку PyTorch; "
                 "MPS требует совместимый Mac и нативный запуск вне Linux-контейнера. "
                 "Для запуска на процессоре явно укажи --device cpu.")
device = torch.device(requested_device)
# Новые данные И параметры моделей создаются на одном выбранном устройстве.
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device, "| dtype: float32")

def make_world(rows, cols, walls=None, items=None):
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise ValueError('N и M должны быть целыми числами.')
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError('Поле должно содержать хотя бы две клетки.')
    walls, items = set(walls or ()), dict(items or {})
    start, goal = (0, 0), (rows - 1, cols - 1)
    inside = lambda p: 0 <= p[0] < rows and 0 <= p[1] < cols
    if any(not inside(p) for p in walls | set(items)):
        raise ValueError('Стена или предмет находится за границей поля.')
    if start in walls or goal in walls:
        raise ValueError('Старт и выход должны быть свободны.')
    if any(p in walls or p in (start, goal) for p in items):
        raise ValueError('Предметы должны лежать вне стен, старта и выхода.')
    if len(items) > 6 or (rows * cols - len(walls) - 1) * 2**len(items) > 2048:
        raise ValueError('Для учебного опыта уменьши поле или число предметов.')
    if any(not -1e6 < value < 1e6 for name, value in items.values()):
        raise ValueError('Стоимость предмета должна быть конечным числом.')
    reached, queue = {start}, [start]
    for row, col in queue:
        if (row, col) == goal:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if inside(nxt) and nxt not in walls and nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    if goal not in reached or any(p not in reached for p in items):
        raise ValueError('Из старта должны быть достижимы выход и все предметы.')
    return dict(rows=rows, cols=cols, start=start, goal=goal, walls=walls,
                items=sorted(items.items()))


def step(world, state, action):
    row, col, collected = state
    if action not in range(4):
        raise ValueError('Действия: 0 вверх, 1 вниз, 2 влево, 3 вправо.')
    if (row, col) == world['goal']:
        return state, 0.0, True
    dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
    nxt = (row + dr, col + dc)
    if not (0 <= nxt[0] < world['rows'] and 0 <= nxt[1] < world['cols']):
        return state, -0.1, False
    if nxt in world['walls']:
        return state, -0.1, False
    reward = -0.04
    for index, (position, (name, value)) in enumerate(world['items']):
        bit = 1 << index
        if nxt == position and not collected & bit:
            reward += value
            collected |= bit
    done = nxt == world['goal']
    if done:
        reward += 2.0
    return (*nxt, collected), reward, done


def train(world, updates=1400):
    states = [(r, c, mask) for mask in range(2**len(world['items']))
              for r in range(world['rows']) for c in range(world['cols'])
              if (r, c) not in world['walls'] and (r, c) != world['goal']]
    lookup = {state: index for index, state in enumerate(states)}
    transitions = [step(world, state, action) for state in states for action in range(4)]

    # Меняем форму массива, сохраняя число и порядок элементов.
    # Создаём тензор — числовой массив — из указанных значений.
    next_ids = torch.tensor([lookup.get(s, 0) for s, _, _ in transitions]).reshape(-1, 4)
    rewards = torch.tensor([r for _, r, _ in transitions]).reshape(-1, 4)
    alive = torch.tensor([not done for _, _, done in transitions]).reshape(-1, 4)

    # Один вход на каждое состояние, четыре выхода — оценки действий Q(s, a).
    features = torch.eye(len(states))

    # Линейный слой учится преобразованию y = x @ W.T + b.
    model = nn.Linear(len(states), 4, bias=False)
    nn.init.zeros_(model.weight)

    # Adam подбирает величину шага для каждого параметра по истории градиентов.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)
    for update in range(updates):
        if update % 20 == 0:
            # Цель заморожена на 20 обновлений; через неё градиенты не проходят.
            with torch.no_grad():
                future = model(features).max(dim=1).values[next_ids]
                target = rewards + 0.95 * future * alive

        # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
        loss = F.mse_loss(model(features), target)

        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)

        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()

        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()
    with torch.no_grad():
        scores = model(features).tolist()
    policy = {s: max(range(4), key=lambda a: scores[i][a]) for i, s in enumerate(states)}

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    return policy, float(loss.detach())


def rollout(world, choose_action, start=None):
    state = (*(start or world['start']), 0)
    route, actions, total, discounted, done = [state], [], 0.0, 0.0, False
    for t in range(world['rows'] * world['cols'] * 8):
        action = choose_action(state)
        state, reward, done = step(world, state, action)
        actions.append(action)
        route.append(state)
        total += reward
        discounted += 0.95**t * reward
        if done:
            break
    return dict(route=route, actions=actions, total=total, discounted=discounted,
                done=done, collected=state[2])


def show(world, result):
    visited = {s[:2] for s in result['route']}
    symbols = {p: str(i + 1) for i, (p, _) in enumerate(world['items'])}
    print('A — старт; G — выход; # — стена; * — маршрут; цифры — предметы.')
    for r in range(world['rows']):
        print(' '.join('A' if (r, c) == world['start'] else
                       'G' if (r, c) == world['goal'] else
                       '#' if (r, c) in world['walls'] else
                       symbols.get((r, c), '*' if (r, c) in visited else '.')
                       for c in range(world['cols'])))
    for i, (position, (name, value)) in enumerate(world['items']):
        print(f'{i + 1}: {name}, стоимость {value}, клетка {position}, '
              f'собран: {bool(result["collected"] & (1 << i))}')
    arrows = ('↑', '↓', '←', '→')
    print('Действия:', ' '.join(arrows[a] for a in result['actions']))
    print('Маршрут:', [s[:2] for s in result['route']])
    print('Выход достигнут:', result['done'], '| шагов:', len(result['actions']))
    print('Сумма наград:', round(result['total'], 3),
          '| с учётом скидки:', round(result['discounted'], 3))


def main():
    import random
    world = demo_world(N, M)
    print(f'Поле {N}×{M}; 0 ↑, 1 ↓, 2 ←, 3 →; скидка будущей награды 0.95.')
    rng = random.Random(7)
    baseline = [rollout(world, lambda state: rng.randrange(4)) for _ in range(100)]
    print('Случайный агент: успехов', sum(r['done'] for r in baseline), '/ 100;',
          'средняя награда со скидкой', round(sum(r['discounted'] for r in baseline) / 100, 3))
    policy, loss = train(world)
    result = rollout(world, lambda state: policy[state])
    print('Ошибка обучения:', round(loss, 6))
    show(world, result)
    assert result['done'], 'Агент не дошёл до выхода; проверь обучение и карту.'

# Для этой расстановки нужны N >= 4 и M >= 5.
N, M = 4, 5

def demo_world(rows, cols):
    if rows < 4 or cols < 5:
        raise ValueError('Для готовой карты со стенами нужны N >= 4 и M >= 5.')
    return make_world(rows, cols, walls={(1, 1), (1, 2), (2, 3)}, items={
        (0, cols - 1): ('Монета', 1.0),
        (2, 1): ('Алмаз', 5.0),
        (rows - 1, 0): ('Сундук', 9.0),
    })

if __name__ == '__main__':
    main()
