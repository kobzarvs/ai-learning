# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

def lookup_limit():
    return {'value': 20, 'unit': 'MB', 'source': 'настройки проекта'}
tools = {'lookup_limit': lookup_limit}

# План в этом опыте задан явно, чтобы видеть границу действий.
plan = [{'tool': 'lookup_limit'}]
observations = []
for step in plan[:3]:
    name = step['tool']
    if name not in tools:
        raise ValueError('Неизвестный инструмент')
    observations.append(tools[name]())
fact = observations[-1]
print(f"Лимит: {fact['value']} {fact['unit']}. Источник: {fact['source']}.")
