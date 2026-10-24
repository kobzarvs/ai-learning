# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.


# Найти нужный фрагмент собственной базы и вернуть ответ вместе с источником.
documents = ['лимит загрузки файла 20 мегабайт', 'пароль можно изменить в настройках',
             'экспорт заметок доступен в меню холста']
vocab = {w:i for i,w in enumerate(sorted(set(' '.join(documents).split())))}
def vector(text):
    result = torch.zeros(len(vocab))
    for word in text.split():
        if word in vocab: result[vocab[word]] += 1
    return F.normalize(result, dim=0)
query = 'какой лимит загрузки файла'
scores = torch.stack([vector(d) for d in documents])@vector(query)
best = int(scores.argmax())
if scores[best] < 0.2:
    print('В базе нет достаточного основания для ответа.')
else:
    print('Найденный фрагмент:', documents[best])
    print('Источник: документ', best+1)
