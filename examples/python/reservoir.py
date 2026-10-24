# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
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

# Создаём последовательность чисел с постоянным шагом; конечная граница не включается.
signal = torch.sin(torch.arange(260)*0.13)

# Создаём искусственные данные из нормального распределения со средним 0.
w = torch.randn(24, 24)*0.08
input_weights = torch.randn(24)*0.3

# Создаём тензор заданной формы и заполняем его нулями.
state, states = torch.zeros(24), []
for value in signal[:-1]:
    state = (w @ state + input_weights*value).tanh()
    states.append(state)

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
x, y = torch.stack(states), signal[1:, None]

# Считаем веса выхода по учебной части; малый штраф стабилизирует решение.
a, b = x[20:200], y[20:200]

# Решаем систему A @ x = b без явного вычисления обратной матрицы.
# Создаём единичную матрицу: единицы на диагонали, остальные элементы — нули.
readout = torch.linalg.solve(a.T@a + 0.001*torch.eye(24), a.T@b)

# MSE — среднее значение квадратов разностей прогноза и правильного ответа.
print('Ошибка на более позднем участке:', F.mse_loss(x[200:]@readout, y[200:]).item())
