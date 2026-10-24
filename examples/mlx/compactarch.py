# Python 3.14 · MLX 0.32.4
import argparse
import platform
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

parser = argparse.ArgumentParser(description="AI Atlas / MLX")
parser.add_argument("--device", choices=["cpu", "metal", "auto"], default="cpu")
requested = parser.parse_args().device
metal_available = platform.system() == "Darwin" and mx.device_count(mx.gpu) > 0
if requested == "auto":
    requested = "metal" if metal_available else "cpu"
if requested == "metal" and not metal_available:
    parser.error("Metal недоступен. Запусти нативно на Apple silicon вне Linux-контейнера либо явно выбери --device cpu.")
mx.set_default_device(mx.gpu if requested == "metal" else mx.cpu)
mx.random.seed(7)
print("MLX | устройство:", requested, "| dtype: float32")

# Создаём искусственные данные из нормального распределения со средним 0.
x = mx.random.normal((1,24,24,16))
# Свёртка применяет обучаемые фильтры к небольшим участкам изображения.
regular = nn.Conv2d(16,32,3,padding=1,bias=False)
# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
compact = nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16,bias=False), nn.Conv2d(16,32,1,bias=False))
counts = []
for name, model in [("Обычная", regular), ("Компактная", compact)]:
    # Разворачиваем дерево параметров модели в список пар «путь, массив».
    count = sum(p.size for _, p in tree_flatten(model.parameters()))
    answer = model(x)
    # MLX откладывает расчёт до запроса результата.
    # Здесь действительно выполняем накопленные вычисления.
    mx.eval(answer)
    counts.append(count)
    print(name, "весов:", count, "байт:", count*4, "ответ NHWC:", answer.shape)
assert counts == [4608, 656]
