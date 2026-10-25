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

# Создаём тензор — числовой массив — из указанных значений.
true_chance = torch.tensor([0.2, 0.5, 0.8])

# Создаём тензор заданной формы и заполняем его единицами.
success = torch.ones(3)
failure = torch.ones(3)

# Создаём тензор заданной формы и заполняем его нулями.
count = torch.zeros(3, dtype=torch.long)
for _ in range(800):
    # Бета-распределение описывает неопределённость вероятности от 0 до 1.
    guesses = torch.distributions.Beta(success, failure).sample()
    action = int(guesses.argmax())

    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    reward = int(torch.rand(()) < true_chance[action])
    success[action] += reward
    failure[action] += 1-reward
    count[action] += 1
print('Сколько раз выбрано каждое действие:', count.tolist())
print('Оценки вероятностей:', (success/(success+failure)).tolist())
