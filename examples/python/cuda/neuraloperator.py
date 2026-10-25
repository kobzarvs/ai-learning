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
x = torch.randn(180, 32)

# Переходим от действительного сигнала к его частотным компонентам.
freq = torch.fft.rfft(x)

# Создаём последовательность чисел с постоянным шагом; конечная граница не включается.
response = torch.exp(-torch.arange(freq.shape[1]).float()/4)

# Восстанавливаем действительный сигнал из частотных компонент.
y = torch.fft.irfft(freq*response, n=32)

# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
# Создаём тензор заданной формы и заполняем его единицами.
weights = nn.Parameter(torch.ones(freq.shape[1], dtype=torch.complex64))

# Adam подбирает величину шага для каждого параметра по истории градиентов.
optimizer = torch.optim.Adam([weights], lr=0.05)
for _ in range(180):
    prediction = torch.fft.irfft(torch.fft.rfft(x[:140])*weights, n=32)

    # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
    loss = F.mse_loss(prediction, y[:140])

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    prediction = torch.fft.irfft(torch.fft.rfft(x[140:])*weights, n=32)
print('Ошибка на новых сигналах:', F.mse_loss(prediction, y[140:]).item())
