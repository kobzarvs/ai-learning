# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

# Создаём тензор — числовой массив — из указанных значений.
current = torch.tensor([0.4]*20)
voltage, spikes = torch.tensor(0.), []
for value in current:
    voltage = 0.9*voltage+value
    spike = (voltage>=1).float()
    voltage = voltage-spike
    spikes.append(int(spike))
print('Импульсы:', spikes)
print('Всего импульсов:', sum(spikes))
