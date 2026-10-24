# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Собрать ограниченный цикл «выбрать инструмент → получить наблюдение → ответить».
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
