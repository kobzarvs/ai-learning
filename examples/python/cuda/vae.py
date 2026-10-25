# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

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
# Создаём тензор — числовой массив — из указанных значений.
x = torch.randn(256, 2)*0.3 + torch.tensor([1.,-1.])

# Линейный слой учится преобразованию y = x @ W.T + b.
encoder = nn.Linear(2, 4)
decoder = nn.Linear(2, 2)

# AdamW обновляет параметры по градиентам.
# Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=0.03)
for _ in range(200):
    mu, logvar = encoder(x).chunk(2, dim=1)
    logvar = logvar.clamp(-8, 8)

    # Создаём случайный шум из нормального распределения той же формы, что исходный
    # тензор.
    z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    reconstruction = F.mse_loss(decoder(z), x)
    kl = 0.5*(mu.square()+logvar.exp()-1-logvar).mean()
    loss = reconstruction + 0.1*kl

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    print('Новые точки:', decoder(torch.randn(5, 2)).tolist())
    print('Ошибка восстановления:', reconstruction.item())
