# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

SUPPORTED_DEVICES = ['cpu', 'cuda', 'mps']
DEVICE_NOTE = 'GPU требует поддержки операций этого опыта установленной версией фреймворка. Автоматический переход на CPU при ошибке отключён.'

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

# weather — условный показатель погоды в 10 000 искусственных наблюдений.
weather = torch.randn(10000)
# ice_cream — условное потребление мороженого; оно зависит от погоды и шума.
ice_cream = weather+0.2*torch.randn(10000)
# swimming — активность купальщиков; здесь она зависит только от погоды и шума.
swimming = 2*weather+0.2*torch.randn(10000)
# Вычисляем корреляции Пирсона между строками матрицы.
# Числа от -1 до 1 показывают силу и направление линейной связи.
# Элемент [0, 1] — связь первого ряда со вторым, а не доказательство причины.
# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
correlation = torch.corrcoef(torch.stack([ice_cream, swimming]))[0,1]
# intervention — принудительно заданное потребление мороженого:
# во всех наблюдениях устанавливаем 100, независимо от погоды.
intervention = torch.full_like(ice_cream, 100.)
# swimming_after — активность купальщиков после такого вмешательства.
# В исходной формуле нет потребления мороженого, поэтому заменять в ней нечего.
swimming_after = 2*weather+0.2*torch.randn(10000)
print('Корреляция:', correlation.item())
print('Изменение среднего купания при вмешательстве:', (swimming_after-swimming).mean().item())
assert intervention.mean() > ice_cream.mean()
