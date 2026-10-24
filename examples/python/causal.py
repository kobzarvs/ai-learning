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

# 1. Создаём наблюдения. weather — условный показатель погоды.
observation_count = 1000

# Создаём искусственные данные из нормального распределения со средним 0.
weather = torch.randn(observation_count)

# Потребление мороженого зависит от погоды и небольших случайных помех.
ice_cream_noise = 0.2 * torch.randn(observation_count)
ice_cream = weather + ice_cream_noise

# Сохраним эти помехи: при сравнении двух условий они будут одинаковыми.
swimming_noise = 0.2 * torch.randn(observation_count)

# direct_effect — прямое влияние мороженого на активность купальщиков.
# 0.0 означает отсутствие такого влияния. Во втором опыте поставь 0.3.
direct_effect = 0.0

def swimming_model(consumption):
    # consumption — потребление мороженого, которое передали в модель.
    weather_effect = 2 * weather
    ice_cream_effect = direct_effect * consumption
    return weather_effect + ice_cream_effect + swimming_noise

# Получаем активность купальщиков при обычном потреблении мороженого.
swimming = swimming_model(ice_cream)

# 2. stack объединяет два ряда в таблицу формы (2, 1000).
# Строка 0 — мороженое, строка 1 — купание; столбец — одно наблюдение.
observed_rows = torch.stack([ice_cream, swimming])

# corrcoef строит таблицу 2 × 2: корреляцию каждой строки с каждой.
# Числа от -1 до 1 показывают силу и направление линейной связи.
correlation_matrix = torch.corrcoef(observed_rows)

# Из таблицы берём связь строки 0 со строкой 1: мороженого с купанием.
correlation = correlation_matrix[0, 1]
correlation_value = correlation.item()
print('Корреляция мороженого и купания:', correlation_value)

# 3. Принудительно задаём потребление равным 3 во всех наблюдениях.
# intervention — массив с этим новым потреблением, независимо от погоды.
intervention_level = 3.0

# Создаём тензор той же формы и заполняем его одной заданной величиной.
intervention = torch.full_like(ice_cream, intervention_level)

# Передаём новое потребление в ту же модель.
# Погода и сохранённые помехи остаются прежними.
swimming_after = swimming_model(intervention)

# 4. Сначала находим изменение каждого ответа, затем среднее изменение.
individual_changes = swimming_after - swimming
average_change = individual_changes.mean()
average_change_value = average_change.item()
print('Среднее изменение активности:', average_change_value)

# По известной формуле эффект равен коэффициенту × изменению потребления.
consumption_changes = intervention - ice_cream
expected_changes = direct_effect * consumption_changes

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
assert torch.allclose(individual_changes, expected_changes, atol=1e-6)
