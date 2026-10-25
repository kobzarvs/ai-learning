# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn

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
data = torch.randn(300)*0.5+3

# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
# Создаём тензор — числовой массив — из указанных значений.
shift = nn.Parameter(torch.tensor(0.))
log_scale = nn.Parameter(torch.tensor(0.))

# Adam подбирает величину шага для каждого параметра по истории градиентов.
optimizer = torch.optim.Adam([shift, log_scale], lr=0.05)
for _ in range(300):
    z = (data-shift)*torch.exp(-log_scale)

    # Минус логарифм плотности с поправкой на изменение масштаба.
    loss = (0.5*z.square()+log_scale).mean()

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()
print('Сдвиг:', shift.item(), 'масштаб:', log_scale.exp().item())
z = torch.randn(5)
samples = shift + log_scale.exp()*z
recovered = (samples-shift)/log_scale.exp()

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
print('Обратимость:', torch.allclose(z, recovered, atol=1e-5))
