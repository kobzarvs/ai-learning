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
x = torch.tensor([[-1.], [0.], [1.]])
y = x.sin()

# Создаём равномерную сетку между двумя границами, включая обе границы.
query = torch.linspace(-3, 3, 21)[:, None]
def kernel(a, b):
    # Вычисляем расстояния между каждой парой объектов из двух наборов.
    return torch.exp(-0.5*torch.cdist(a, b).square())

# Создаём единичную матрицу: единицы на диагонали, остальные элементы — нули.
k = kernel(x, x)+0.01*torch.eye(len(x))
cross = kernel(query, x)

# Решаем систему A @ x = b без явного вычисления обратной матрицы.
mean = cross @ torch.linalg.solve(k, y)

# Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).
variance = (1-(cross*torch.linalg.solve(k, cross.T).T).sum(1)).clamp_min(0)
for i in [0, 10, 20]:
    print('x =', query[i].item(), 'среднее =', mean[i].item(), 'разброс =', variance[i].sqrt().item())
