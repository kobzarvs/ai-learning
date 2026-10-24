# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_DEVICES = ['cpu', 'cuda']
DEVICE_NOTE = 'Здесь используются линейная алгебра, комплексные FFT или специальные распределения; вариант MPS пока не включён.'

import argparse

parser = argparse.ArgumentParser(description="Учебный пример AI Atlas / PyTorch")
parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu",
                    help="cpu; NVIDIA cuda; Apple Metal mps; auto выбирает доступный GPU")
requested_device = parser.parse_args().device
available = {"cpu": True, "cuda": torch.cuda.is_available(),
             "mps": torch.backends.mps.is_available()}
if requested_device == "auto":
    requested_device = next(d for d in ("cuda", "mps", "cpu")
                            if d in SUPPORTED_DEVICES and available[d])
if requested_device not in SUPPORTED_DEVICES:
    parser.error("Для этого опыта доступны: " + ", ".join(SUPPORTED_DEVICES)
                 + ". " + DEVICE_NOTE)
if not available[requested_device]:
    parser.error("Устройство " + requested_device + " недоступно. "
                 "CUDA требует NVIDIA, драйвер и CUDA-сборку PyTorch; "
                 "MPS требует совместимый Mac и нативный запуск вне Linux-контейнера. "
                 "Для запуска на процессоре явно укажи --device cpu.")
device = torch.device(requested_device)
# Новые данные И параметры моделей создаются на одном выбранном устройстве.
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device, "| dtype: float32")

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
