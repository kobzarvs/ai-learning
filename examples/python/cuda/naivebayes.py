# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch

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

# Соединяем тензоры вдоль уже существующего измерения dim.
# Создаём искусственные данные из нормального распределения со средним 0.
x = torch.cat([torch.randn(120, 2)-1, torch.randn(120, 2)+1])

# Создаём тензор заданной формы и заполняем его нулями.
# Создаём тензор заданной формы и заполняем его единицами.
y = torch.cat([torch.zeros(120), torch.ones(120)]).long()

# Создаём случайную перестановку индексов для перемешивания объектов.
order = torch.randperm(len(x))
x, y = x[order], y[order]
means, variances, priors = [], [], []
for label in range(2):
    subset = x[:180][y[:180] == label]
    means.append(subset.mean(0))
    variances.append(subset.var(0)+1e-4)
    priors.append(len(subset)/180)

# Собираем тензоры одинаковой формы в один, добавляя новое измерение.
mu, var = torch.stack(means), torch.stack(variances)
log_score = -0.5*((x[180:, None]-mu).square()/var+var.log()).sum(2)

# Создаём тензор — числовой массив — из указанных значений.
log_score += torch.tensor(priors).log()
print('Верно:', log_score.argmax(1).eq(y[180:]).float().mean().item())
