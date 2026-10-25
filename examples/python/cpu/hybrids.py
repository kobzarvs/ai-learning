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

# Модель предлагает действия: читать, отправить, удалить.
logits = torch.tensor([[1., 2., 5.]])
allowed = torch.tensor([[True, True, False]])

# Заменяем элементы, отмеченные логической маской, указанным значением.
restricted = logits.masked_fill(~allowed, float('-inf'))

# Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
probabilities = restricted.softmax(1)
action = probabilities.argmax(1).item()
print('Вероятности допустимых действий:', probabilities.tolist())
print('Выбрано действие:', ['читать', 'отправить', 'удалить'][action])
assert probabilities[0, 2] == 0
