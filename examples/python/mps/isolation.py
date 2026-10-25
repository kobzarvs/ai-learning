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

# Соединяем тензоры вдоль уже существующего измерения dim.
# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.cat([torch.randn(100, 2)*0.3, torch.tensor([[5.,5.]])])
def depth(values, index, level=0):
    if len(values)<=1 or level>=30:
        return level

    # Генерируем случайные целые числа; верхняя граница диапазона не включается.
    feature = int(torch.randint(0, values.shape[1], (1,)))
    low, high = values[:, feature].min(), values[:, feature].max()
    if high == low:
        return level

    # Генерируем случайные числа, равномерно распределённые от 0 до 1.
    cut = low + torch.rand(())*(high-low)
    side = values[:, feature] < cut
    keep = side == side[index]
    new_index = int(keep[:index].sum())
    return depth(values[keep], new_index, level+1)
for index in [0, 100]:
    paths = [depth(x, index) for _ in range(100)]
    print('Точка', index, 'средняя глубина:', sum(paths)/len(paths))
