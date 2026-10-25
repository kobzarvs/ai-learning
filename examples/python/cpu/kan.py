# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cpu")

# Новые тензоры и параметры модели создаются на выбранном устройстве.
torch.set_default_device(device)

# float32 — вещественные числа по 32 бита на значение.
torch.set_default_dtype(torch.float32)

# Начальное состояние генератора позволяет повторить случайные данные.
torch.manual_seed(7)
torch.set_num_threads(1)
print("PyTorch", torch.__version__, "| устройство:", device)

def fit_regression(model, x, y, steps=200):
    # AdamW обновляет параметры по градиентам.
    # Параметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)

        # MSE — среднее значение квадратов разностей прогноза и правильного ответа.
        loss = F.mse_loss(prediction, y)

        # Вычисляем градиенты ошибки по обучаемым параметрам.
        loss.backward()

        # Обновляем обучаемые параметры по рассчитанным градиентам.
        optimizer.step()
    model.eval()

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    return float(loss.detach())

# Генерируем случайные числа, равномерно распределённые от 0 до 1.
x = torch.rand(400, 2) * 2 - 1
y = (x[:, :1].sin() + x[:, 1:].square())
class EdgeFunctions(nn.Module):
    def __init__(self):
        super().__init__()

        # Сохраняем тензор внутри модели без включения в обучаемые параметры.
        # Создаём равномерную сетку между двумя границами, включая обе границы.
        self.centers = nn.Buffer(torch.linspace(-1, 1, 12))

        # Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
        # Создаём тензор заданной формы и заполняем его нулями.
        self.coefficients = nn.Parameter(torch.zeros(2, 12))
    def forward(self, x):
        bases = torch.exp(-((x[..., None]-self.centers)/0.25).square())
        return (bases*self.coefficients).sum((1, 2)).unsqueeze(1)
model = EdgeFunctions()
fit_regression(model, x[:320], y[:320])

# Не записываем эти операции в граф для автоматического вычисления градиентов.
with torch.no_grad():
    prediction = model(x[320:])
    test_error = F.mse_loss(prediction, y[320:])
    print('Ошибка на новых точках:', test_error.item())
