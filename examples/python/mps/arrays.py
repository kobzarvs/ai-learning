# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

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

# Создаём тензор — числовой массив — из указанных значений.
x = torch.tensor([[1.,2.,3.], [4.,5.,6.]])
weights = torch.tensor([[1.,0.], [0.,1.], [1.,1.]])
answer = x@weights
print('Вход:', tuple(x.shape), 'веса:', tuple(weights.shape))
print('Ответ:', tuple(answer.shape), answer.tolist())
assert answer.tolist() == [[4.,5.], [10.,11.]]
