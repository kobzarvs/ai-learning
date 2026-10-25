# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch.nn import functional as F

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS недоступен: нужен совместимый Mac и нативный запуск.")

device = torch.device("mps")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

documents = ['лимит загрузки файла 20 мегабайт', 'пароль можно изменить в настройках',
             'экспорт заметок доступен в меню холста']
vocab = {w:i for i,w in enumerate(sorted(set(' '.join(documents).split())))}
def vector(text):
    # Создаём тензор заданной формы и заполняем его нулями.
    result = torch.zeros(len(vocab))
    for word in text.split():
        if word in vocab: result[vocab[word]] += 1

    # Нормируем векторы; по умолчанию их евклидова длина становится равной 1.
    return F.normalize(result, dim=0)
query = 'какой лимит загрузки файла'

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
scores = torch.stack([vector(d) for d in documents])@vector(query)
best = int(scores.argmax())
if scores[best] < 0.2:
    print('В базе нет достаточного основания для ответа.')
else:
    print('Найденный фрагмент:', documents[best])
    print('Источник: документ', best+1)
