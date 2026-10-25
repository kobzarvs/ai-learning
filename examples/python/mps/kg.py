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

facts = [('камера', 'часть', 'робот'), ('робот', 'находится', 'цех'),
         ('датчик', 'часть', 'станок'), ('станок', 'находится', 'склад')]
object_name = 'камера'
parents = [b for a, relation, b in facts if a==object_name and relation=='часть']
places = [b for a, relation, b in facts if a in parents and relation=='находится']
print('Где находится камера:', places)
print('Основание:', [f for f in facts if f[0] in [object_name]+parents])
assert places == ['цех']
