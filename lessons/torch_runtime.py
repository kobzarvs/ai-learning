"""Embedded in each exported example, so downloaded scripts remain standalone."""

RUNTIME = '''
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
'''

# These examples use operators with limited MPS support. Keep the advertised
# choices conservative rather than silently moving part of the work to CPU.
MPS_LIMITED = {
    'reservoir', 'gp', 'decomposition', 'factorweights', 'symbolicreg',
    'neuraloperator', 'bayes', 'probability', 'bandits',
}

def device_info(topic):
    if topic in MPS_LIMITED:
        return ['cpu', 'cuda'], 'Здесь используются линейная алгебра, комплексные FFT или специальные распределения; вариант MPS пока не включён.'
    return ['cpu', 'cuda', 'mps'], 'GPU требует поддержки операций этого опыта установленной версией фреймворка. Автоматический переход на CPU при ошибке отключён.'

def runtime(topic):
    devices, note = device_info(topic)
    return '\nSUPPORTED_DEVICES = ' + repr(devices) + '\nDEVICE_NOTE = ' + repr(note) + '\n' + RUNTIME
