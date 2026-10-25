# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна: проверь видеокарту NVIDIA и установку PyTorch.")

device = torch.device("cuda")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
print("PyTorch", torch.__version__, "| устройство:", device)

# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.randn(300, 3)

# Более поздние данные пришли из других условий.
x[240:] += 1.5
train, validation, test = x[:180], x[180:240], x[240:]
mean = train.mean(0)

# Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).
std = train.std(0).clamp_min(1e-6)
train_scaled = (train-mean)/std
test_scaled = (test-mean)/std
print('Среднее учебной части:', train_scaled.mean(0).tolist())
print('Среднее теста с учебными настройками:', test_scaled.mean(0).tolist())
assert train_scaled.mean(0).abs().max()<1e-6
