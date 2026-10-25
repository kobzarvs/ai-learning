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

# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
# Создаём тензор заданной формы и заполняем его нулями.
logits = nn.Parameter(torch.zeros(2))

# Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
# Устойчиво вычисляем логарифмы вероятностей классов из исходных оценок.
reference = torch.zeros(2).log_softmax(0).detach()

# Adam подбирает величину шага для каждого параметра по истории градиентов.
optimizer = torch.optim.Adam([logits], lr=0.05)
for _ in range(100):
    logp = logits.log_softmax(0)

    # В каждой паре ответ 1 предпочитается ответу 0.
    advantage = (logp[1]-logp[0])-(reference[1]-reference[0])

    # Устойчиво вычисляем логарифм сигмоиды, не вычисляя её отдельно.
    loss = -F.logsigmoid(0.5*advantage)

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
print('Вероятности ответов:', logits.softmax(0).detach().tolist())
