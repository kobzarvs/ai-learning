# Python 3.14 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import torch
from torch import nn

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

# Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.
# Создаём тензор заданной формы и заполняем его нулями.
logits = nn.Parameter(torch.zeros(3))

# Adam подбирает величину шага для каждого параметра по истории градиентов.
optimizer = torch.optim.Adam([logits], lr=0.04)

# Создаём тензор — числовой массив — из указанных значений.
reward_table = torch.tensor([0., 0.4, 1.])
for _ in range(250):
    # Задаём случайный выбор одного действия по вероятностям категорий.
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample((64,))
    rewards = reward_table[actions]
    advantage = rewards-rewards.mean()

    # Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.
    loss = -(distribution.log_prob(actions)*advantage.detach()).mean()

    # Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.
    optimizer.zero_grad(set_to_none=True)

    # Вычисляем градиенты ошибки по обучаемым параметрам.
    loss.backward()

    # Обновляем обучаемые параметры по рассчитанным градиентам.
    optimizer.step()

# Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.
print('Вероятности действий:', logits.softmax(0).detach().tolist())
