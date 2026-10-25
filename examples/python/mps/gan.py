# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
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

# Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.
# Линейный слой учится преобразованию y = x @ W.T + b.
generator = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
critic = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
g_opt = torch.optim.AdamW(generator.parameters(), lr=0.01)
d_opt = torch.optim.AdamW(critic.parameters(), lr=0.01)
for _ in range(300):
    # Создаём искусственные данные из нормального распределения со средним 0.
    real = torch.randn(64, 1)*0.4+2
    fake = generator(torch.randn(64, 1))

    # Считаем ошибку бинарного классификатора прямо по исходным оценкам.
    d_loss = F.binary_cross_entropy_with_logits(critic(real), torch.ones_like(real))

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    d_loss += F.binary_cross_entropy_with_logits(critic(fake.detach()), torch.zeros_like(fake))

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    d_opt.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    d_loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    d_opt.step()

    # Генератор получает градиент через критика, но не меняет его веса.
    critic.requires_grad_(False)
    g_loss = F.binary_cross_entropy_with_logits(critic(fake), torch.ones_like(fake))
    g_opt.zero_grad(set_to_none=True)
    g_loss.backward()
    g_opt.step()
    critic.requires_grad_(True)

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    generated = generator(torch.randn(1000, 1))
print('Среднее и разброс:', generated.mean().item(), generated.std().item())
