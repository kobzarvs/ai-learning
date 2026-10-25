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

# 1. Один сотрудник выполняет одно задание. Сложность — от 1 до 5.
employee_count = 1000

# Генерируем случайные числа, равномерно распределённые от 0 до 1.
task_difficulty = 1 + 4 * torch.rand(employee_count)

# Сохраняем случайные числа для назначения инструкции и помехи времени.
assignment_draw = torch.rand(employee_count)
time_noise = 2 * torch.rand(employee_count) - 1

# minutes_saved — сколько минут новая инструкция экономит на каждом задании.
minutes_saved = 4.0

# False: новую инструкцию чаще дают на сложных заданиях.
# True: назначают случайно, с вероятностью 50% для каждого сотрудника.
random_assignment = False
if random_assignment:
    # full_like создаёт массив той же формы, заполняя его заданным числом.
    assignment_probability = torch.full_like(task_difficulty, 0.5)
else:
    # Вероятность растёт от 10% для простых заданий до 90% для сложных.
    assignment_probability = 0.1 + 0.2 * (task_difficulty - 1)

# Сравнение даёт массив True/False: кому досталась новая инструкция.
uses_new_instruction = assignment_draw < assignment_probability

# float переводит True/False в числа: 1 — новая инструкция, 0 — прежняя.
instruction_flag = uses_new_instruction.float()

def completion_time(instruction):
    # instruction — массив нулей и единиц, задающий инструкцию для каждого.
    task_time = 10 * task_difficulty
    time_reduction = minutes_saved * instruction
    return 20 + task_time - time_reduction + time_noise

# 2. Вычисляем время при фактически назначенных инструкциях.
observed_time = completion_time(instruction_flag)

# stack собирает таблицу (2, 1000): строка 0 — инструкция, строка 1 — время.
# Каждый столбец соответствует одному сотруднику с его заданием.
observed_rows = torch.stack([instruction_flag, observed_time])

# corrcoef строит таблицу 2 × 2: корреляцию каждой строки с каждой.
# Значения от -1 до 1 описывают направление и силу линейной связи.
correlation_matrix = torch.corrcoef(observed_rows)

# Берём связь применения новой инструкции со временем выполнения.
correlation = correlation_matrix[0, 1]

# item извлекает обычное число из тензора с единственным значением.
correlation_value = correlation.item()
print('Корреляция новой инструкции со временем:', correlation_value)

# 3. Выбираем время сотрудников с новой инструкцией и с прежней.
# Знак ~ меняет True на False и наоборот: получаем вторую группу.
new_group_times = observed_time[uses_new_instruction]
old_group_times = observed_time[~uses_new_instruction]

# mean вычисляет среднее время отдельно в каждой группе.
new_group_mean = new_group_times.mean()
old_group_mean = old_group_times.mean()
observed_difference = new_group_mean - old_group_mean
print('Разница между группами, мин:', observed_difference.item())

# 4. Вмешательство: задаём инструкцию всем, независимо от сложности.
# zeros_like и ones_like создают массивы нулей и единиц той же формы.
everyone_old = torch.zeros_like(task_difficulty)
everyone_new = torch.ones_like(task_difficulty)

# Пересчитываем ТЕ ЖЕ задания по обеим инструкциям.
# Сложность и сохранённые помехи времени в обоих расчётах одинаковые.
time_with_old = completion_time(everyone_old)
time_with_new = completion_time(everyone_new)

# Сначала разница для каждого задания, затем средняя разница.
individual_changes = time_with_new - time_with_old
average_change = individual_changes.mean()
print('Причинное изменение, мин:', average_change.item())
print('Отрицательное изменение означает ускорение.')

# allclose проверяет равенство массивов с допуском на округление float32.
expected_changes = torch.full_like(task_difficulty, -minutes_saved)

# Проверяем совпадение всех значений с учётом небольшой ошибки округления.
assert torch.allclose(individual_changes, expected_changes, atol=1e-5)
