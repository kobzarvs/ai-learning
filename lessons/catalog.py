"""Source for runnable lessons. No downloads, tokens, or private datasets."""
from textwrap import dedent
LESSONS = {}

def add(ids, goal, code, expected, exercise, explanation, limits='', kind='Учебный эксперимент'):
    for topic in ids.split():
        if topic in LESSONS:
            raise ValueError('Duplicate lesson: ' + topic)
        LESSONS[topic] = dict(goal=goal, body=dedent(code).strip(), expected=expected,
          exercise=exercise, explanation=explanation, kind=kind,
          limits=limits or 'Данные здесь маленькие и искусственные. Успех на них проверяет идею, но не качество на реальных данных. Перед применением собери отдельный проверочный набор из своей задачи.')

PREFIX = '''# Python 3.13 · PyTorch 2.14.1
# Данные создаются внутри программы; ничего скачивать не нужно.
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)  # Повторяемая случайная инициализация.
torch.set_num_threads(1)  # Для маленького опыта достаточно одного потока CPU.
'''
HELPERS = {
'fit_classifier': '''
def fit_classifier(model, x, y, steps=160):
    # Обучаем только на переданной учебной части данных.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())
''',
'accuracy': '''
@torch.no_grad()
def accuracy(model, x, y):
    # argmax выбирает номер категории с наибольшей оценкой.
    return float((model(x).argmax(dim=-1) == y).float().mean())
''',
'fit_regression': '''
def fit_regression(model, x, y, steps=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())
''',
'points': '''
def points(count=320):
    x = torch.randn(count, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y
''',
'stripes': '''
def stripes(count=240):
    # 0 = вертикальная полоса, 1 = горизонтальная. Остальное — шум.
    y = torch.randint(0, 2, (count,))
    x = torch.randn(count, 1, 8, 8) * 0.12
    for i, label in enumerate(y):
        pos = int(torch.randint(1, 6, (1,)))
        if label == 0:
            x[i, 0, :, pos:pos+2] += 1
        else:
            x[i, 0, pos:pos+2, :] += 1
    return x, y
'''
}

add('mlp', 'Научить сеть отличать точки внутри круга от точек снаружи.', '''
x = torch.rand(500, 2) * 2 - 1
y = (x.square().sum(1) < 0.45).long()
model = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 2))
fit_classifier(model, x[:400], y[:400])
print('Доля верных ответов на новых точках:', accuracy(model, x[400:], y[400:]))
''', 'Доля верных ответов заметно выше случайного выбора между двумя категориями. Единица означает, что все ответы верны.', 'Замени 24 скрытых нейрона на 4, затем на 64. Сравни качество на тех же новых точках.', 'Первый слой смешивает два числа — координаты точки. SiLU изгибает результат, поэтому следующему слою доступна нелинейная граница: круг, а не только прямая.')

add('cnn vision lab-vision', 'Обучить небольшую сеть различать вертикальные и горизонтальные полосы на картинках 8 × 8.', '''
x, y = stripes()
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2),
)
fit_classifier(model, x[:180], y[:180])
print('Верно на новых картинках:', accuracy(model, x[180:], y[180:]))
print('Обучаемых чисел:', sum(p.numel() for p in model.parameters()))
''', 'Сеть должна распознавать большинство новых полос, хотя они сдвинуты и зашумлены. Программа также выводит число параметров.', 'Подними шум с 0.12 до 0.4. Сравни одну свёртку и две; не меняй проверочные картинки между опытами.', 'Свёртка двигает маленькое окно по изображению и ищет один и тот же признак в разных местах. Усреднение собирает признаки всей картинки; последний слой выбирает ориентацию.', 'Это полный учебный цикл обучения маленькой CNN, но полосы проще фотографий. Для реальных фото нужны разнообразные снимки и отдельные данные из других условий.')

add('rnn lstm', 'Прочитать последовательность чисел и предсказать их среднее.', '''
x = torch.randn(300, 12, 1)
y = x.mean(dim=1)
class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.LSTM(1, 16, batch_first=True)
        self.output = nn.Linear(16, 1)
    def forward(self, values):
        states, _ = self.memory(values)
        return self.output(states[:, -1])
model = SequenceModel()
fit_regression(model, x[:240], y[:240])
with torch.no_grad():
    print('Ошибка на новых последовательностях:', F.mse_loss(model(x[240:]), y[240:]).item())
''', 'Средняя квадратичная ошибка должна быть меньше, чем при постоянном ответе 0: для этого набора ориентир около 1/12.', 'Увеличь длину с 12 до 60. Сравни память LSTM с обычным nn.RNN: у обоих одинаковый интерфейс в этом примере.', 'Слой памяти получает по одному числу и обновляет внутреннее состояние. Последнее состояние передаётся линейному слою, который выдаёт одно число. LSTM регулирует, какую информацию сохранить.', 'Это демонстрация последовательной памяти; для новой задачи полезно сравнивать её с современными свёртками, attention и моделями пространства состояний.')

add('tcn audio lab-signal', 'По короткому сигналу отличить медленные колебания от быстрых.', '''
t = torch.linspace(0, 1, 64)
y = torch.randint(0, 2, (300,))
x = torch.sin(2 * math.pi * (3 + 6*y[:, None]) * t)
x = (x + 0.15 * torch.randn_like(x)).unsqueeze(1)
model = nn.Sequential(nn.Conv1d(1, 12, 5, dilation=2), nn.SiLU(),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, 2))
fit_classifier(model, x[:240], y[:240])
print('Доля верных ответов:', accuracy(model, x[240:], y[240:]))
''', 'Сеть различает два диапазона частот на новых сигналах. Значение 0.5 соответствовало бы случайному выбору.', 'Добавь случайную фазу к синусу. Проверь, удержится ли качество, когда сигнал начинается в другом месте волны.', 'Одномерная свёртка ищет рисунок в соседних отсчётах. dilation=2 раздвигает точки окна: оно видит более широкий участок при том же числе весов.', 'В примере показана расширенная свёртка. Для потокового прогноза нужна причинная свёртка: её окно не должно видеть будущие отсчёты. Здесь классифицируется целиком уже записанный сигнал.')

add('transformer', 'Посмотреть, как attention связывает элементы последовательности, и проверить запрет смотреть в будущее.', '''
x = torch.randn(2, 6, 16)  # 2 последовательности, 6 позиций, 16 признаков.
projection = nn.Linear(16, 48, bias=False)
q, k, v = projection(x).chunk(3, dim=-1)
# Измерения: набор, голова, позиция, признаки головы.
q, k, v = [z.view(2, 6, 4, 4).transpose(1, 2) for z in (q, k, v)]
answer = F.scaled_dot_product_attention(q, k, v, is_causal=True)
changed_v = v.clone()
changed_v[:, :, -1] += 100
changed = F.scaled_dot_product_attention(q, k, changed_v, is_causal=True)
print('Размер ответа:', tuple(answer.shape))
print('Ранние ответы не меняются:', torch.allclose(answer[:, :, :-1], changed[:, :, :-1]))
assert torch.allclose(answer[:, :, :-1], changed[:, :, :-1])
''', 'Размер ответа — (2, 4, 6, 4). Проверка выдаёт True: изменение последней позиции не влияет на предыдущие.', 'Выключи is_causal в обоих вызовах. Посмотри, почему проверка перестала выполняться.', 'q задаёт, что ищет каждая позиция; k описывает, что она предлагает другим; v содержит передаваемые сведения. Причинная маска разрешает получать информацию только из текущей и предыдущих позиций.', 'Показан один механизм attention через актуальный API PyTorch. Здесь нет обучения, позиционного кодирования и полной языковой модели.', kind='Демонстрация механизма')

add('gnn graphsdata', 'Передать сообщения между связанными объектами и увидеть, что изолированный объект не получает чужие сведения.', '''
adjacency = torch.tensor([[1.,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]])
features = torch.tensor([[1.,0],[0,1],[1,1],[8,8]])
weights = adjacency / adjacency.sum(1, keepdim=True)
layer = nn.Linear(2, 3, bias=False)
messages = weights @ features
answer = layer(messages)
print('Сведения после обмена:', messages.tolist())
print('Изолированный объект:', messages[-1].tolist())
assert torch.allclose(messages[-1], features[-1])
''', 'Первые три объекта смешивают сведения соседей. Последний остаётся [8, 8], потому что с ним никто не связан.', 'Соедини последний объект с первым: измени оба симметричных элемента матрицы. Посмотри, какие сообщения изменились.', 'Матрица adjacency — список связей в форме таблицы. Единицы на диагонали сохраняют собственные сведения. Умножение на признаки усредняет соседей, а Linear делает обучаемое преобразование.', 'Это один шаг передачи сообщений без обучения. Полноценная GNN требует задачи, правильного разбиения графа и обычно специальной библиотеки для разреженных связей.', kind='Демонстрация механизма')

add('siamese contrastive', 'Научить общий преобразователь выдавать похожие числа для двух зашумлённых видов одного объекта.', '''
x = torch.randn(128, 8)
encoder = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 12))
optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.01)
for _ in range(100):
    a = F.normalize(encoder(x + 0.1*torch.randn_like(x)), dim=1)
    b = F.normalize(encoder(x + 0.1*torch.randn_like(x)), dim=1)
    scores = a @ b.T / 0.1
    loss = F.cross_entropy(scores, torch.arange(len(x)))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    new = torch.randn(40, 8)
    a = F.normalize(encoder(new), dim=1)
    b = F.normalize(encoder(new + 0.1*torch.randn_like(new)), dim=1)
    print('Найден свой второй вид:', (a @ b.T).argmax(1).eq(torch.arange(40)).float().mean().item())
''', 'Большинство новых объектов находят свой второй вид среди 40 вариантов. Случайный результат был бы около 1/40.', 'Увеличь шум с 0.1 до 1.0. Объясни, в какой момент два вида перестают надёжно описывать один объект.', 'Одна сеть используется дважды: это и есть общие веса двух ветвей. Ошибка поощряет совпадение пар и одновременно отталкивает чужие объекты, чтобы сеть не выдавала всем одно число.')

add('specialblocks compactarch tinyml', 'Сравнить число параметров обычной свёртки и разделённой на обработку каналов и их смешивание.', '''
x = torch.randn(1, 16, 24, 24)
regular = nn.Conv2d(16, 32, 3, padding=1, bias=False)
compact = nn.Sequential(
    nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
    nn.Conv2d(16, 32, 1, bias=False),
)
for name, model in [('Обычная', regular), ('Компактная', compact)]:
    count = sum(p.numel() for p in model.parameters())
    print(name, 'параметров:', count, 'байт весов:', count*4,
          'размер ответа:', tuple(model(x).shape))
''', 'Обычный слой содержит 4608 весов, компактный — 656. Оба возвращают (1, 32, 24, 24), но значения ответов и возможности слоёв различаются.', 'Добавь между двумя компактными слоями nn.SiLU(). Посчитай отдельно память промежуточных карт признаков.', 'groups=16 заставляет каждый из 16 каналов обрабатываться отдельно. Свёртка 1 × 1 смешивает каналы. Так устроена одна из идей компактных моделей для устройств.', 'Меньше весов не гарантирует меньшую задержку. Код не измеряет скорость и не доказывает пригодность для микроконтроллера: там важны также память активаций и поддержка операций.', kind='Разбор компактного блока')

add('ssm', 'Проследить, как модель пространства состояний хранит затухающую память о входном импульсе.', '''
x = torch.zeros(30)
x[3] = 1
state = torch.zeros(4)
decay = torch.tensor([0.1, 0.5, 0.8, 0.95])
readout = torch.tensor([0.25, 0.25, 0.25, 0.25])
outputs = []
for value in x:
    state = decay * state + value
    outputs.append((readout * state).sum().item())
print('Ответ после импульса:', [round(v, 3) for v in outputs[3:12]])
assert outputs[3] > outputs[8] > outputs[20] > 0
''', 'После импульса ответ постепенно уменьшается. Компоненты с коэффициентом 0.95 помнят событие дольше, чем с 0.1.', 'Поставь второй импульс на позиции 12. Затем сделай коэффициент памяти зависимым от входа и сравни поведение.', 'Состояние — несколько чисел, которые обновляются на каждом шаге. Разные коэффициенты дают короткую и длинную память; выход собирает их вместе.', 'Это линейная модель состояния, поясняющая основу семейства. Это не реализация Mamba-3: у неё селективные, зависящие от данных переходы и специальные вычислительные ядра. Текущая официальная реализация указана в источниках.', kind='Демонстрация механизма')

add('linearattention', 'Проверить, что линейное attention можно вычислять через накопленную память без таблицы всех пар позиций.', '''
q, k, v = [torch.rand(12, 8) for _ in range(3)]
# Положительные признаки делают знаменатель устойчивым.
phi = lambda x: F.elu(x) + 1
q, k = phi(q), phi(k)
memory, normalizer = torch.zeros(8, 8), torch.zeros(8)
answers = []
for t in range(len(q)):
    memory += torch.outer(k[t], v[t])
    normalizer += k[t]
    answers.append((q[t] @ memory) / (q[t] @ normalizer + 1e-6))
recurrent = torch.stack(answers)
scores = (q @ k.T).tril()
reference = (scores @ v) / (scores.sum(1, keepdim=True) + 1e-6)
print('Максимальная разница:', (recurrent-reference).abs().max().item())
assert torch.allclose(recurrent, reference, atol=1e-5)
''', 'Два пути вычисления совпадают с небольшой погрешностью округления. Память в цикле не растёт с числом уже прочитанных позиций.', 'Увеличь длину до 1000. Сравни размеры memory и scores, не только время на маленьком примере.', 'Вместо хранения всех попарных связей накапливаются сумма произведений ключей на значения и сумма ключей. Запрос читает эту сжатую память.', 'Это положительное kernel attention, а не точная замена softmax и не полная реализация современных gated delta-сетей. Совпадение проверяется с тем же ядром, а не с обычным Transformer.', kind='Демонстрация механизма')

add('moe', 'Направить каждый объект к одному из трёх маленьких экспертов.', '''
x = torch.randn(10, 4)
router = nn.Linear(4, 3)
experts = nn.ModuleList([nn.Linear(4, 2) for _ in range(3)])
scores = router(x).softmax(dim=1)
choice = scores.argmax(dim=1)
answer = torch.zeros(10, 2)
for i, expert in enumerate(experts):
    selected = choice == i
    answer[selected] = expert(x[selected])
print('Эксперт для каждого объекта:', choice.tolist())
print('Объектов у экспертов:', torch.bincount(choice, minlength=3).tolist())
print('Ответов:', tuple(answer.shape))
''', 'У каждого из 10 объектов выбран ровно один эксперт; распределение нагрузки может оказаться неравномерным.', 'Добавь большой положительный сдвиг одному выходу router. Посмотри, почему без балансировки остальные эксперты перестают получать примеры.', 'Маршрутизатор выбирает подходящую сеть, и вычисляется только её ответ. Общее число хранимых весов при этом включает всех экспертов.', 'Показана маршрутизация без обучения. Жёсткий argmax сам не обучает router; в рабочей модели нужны выбранные вероятности в выходе, балансирующая ошибка и стратегия обучения.', kind='Демонстрация механизма')

add('kan', 'Обучить сумму нелинейных функций на рёбрах описывать зависимость sin(x₁) + x₂².', '''
x = torch.rand(400, 2) * 2 - 1
y = (x[:, :1].sin() + x[:, 1:].square())
class EdgeFunctions(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Buffer(torch.linspace(-1, 1, 12))
        self.coefficients = nn.Parameter(torch.zeros(2, 12))
    def forward(self, x):
        bases = torch.exp(-((x[..., None]-self.centers)/0.25).square())
        return (bases*self.coefficients).sum((1, 2)).unsqueeze(1)
model = EdgeFunctions()
fit_regression(model, x[:320], y[:320])
with torch.no_grad():
    print('Ошибка на новых точках:', F.mse_loss(model(x[320:]), y[320:]).item())
''', 'Сеть приближает сумму двух кривых. Снижение ошибки показывает, что обучаемые функции на рёбрах подстроились под данные.', 'Замени целевую функцию на x₁ × x₂. Почему одного такого аддитивного слоя недостаточно?', 'Вместо одного веса у каждого входа есть набор коэффициентов для колоколообразных функций. Их сумма образует гибкую кривую, а затем складываются вклады входов.', 'Это однослойная модель с радиальными базисами, иллюстрирующая идею KAN. Она не повторяет spline-KAN и не подтверждает превосходство над обычной сетью.', kind='Демонстрация механизма')

add('liquid', 'Увидеть, как непрерывное состояние меняется при разных интервалах между наблюдениями.', '''
class ContinuousCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Linear(2, 4)
        self.speed = nn.Linear(2, 4)
    def forward(self, state, x, dt):
        target = self.target(x).tanh()
        decay = torch.exp(-F.softplus(self.speed(x))*dt)
        return decay*state + (1-decay)*target
cell = ContinuousCell()
x, state = torch.ones(1, 2), torch.zeros(1, 4)
for dt in [0.01, 0.1, 1.0]:
    print('Интервал', dt, 'состояние', cell(state, x, dt).detach().tolist())
''', 'При большем интервале состояние сильнее приближается к целевому. Интервал является частью вычисления, а не просто номером строки.', 'Задай dt=0 и проверь, что состояние не меняется. Затем подай другой вход и сравни скорость изменения.', 'Сеть вычисляет, к какому состоянию двигаться и как быстро. Экспонента переводит длительность паузы в долю сохранённой памяти.', 'Это простая непрерывная ячейка, а не точная реализация LTC или CfC. Для их сравнения используй официальную библиотеку ncps.', kind='Демонстрация механизма')

add('reservoir', 'Обучить только выходной слой, оставив случайную рекуррентную память неизменной.', '''
signal = torch.sin(torch.arange(260)*0.13)
w = torch.randn(24, 24)*0.08
input_weights = torch.randn(24)*0.3
state, states = torch.zeros(24), []
for value in signal[:-1]:
    state = (w @ state + input_weights*value).tanh()
    states.append(state)
x, y = torch.stack(states), signal[1:, None]
# Считаем веса выхода по учебной части; малый штраф стабилизирует решение.
a, b = x[20:200], y[20:200]
readout = torch.linalg.solve(a.T@a + 0.001*torch.eye(24), a.T@b)
print('Ошибка на более позднем участке:', F.mse_loss(x[200:]@readout, y[200:]).item())
''', 'Выходной слой предсказывает следующий отсчёт периодического сигнала с небольшой ошибкой на более позднем участке.', 'Увеличь масштаб w с 0.08 до 0.5. Посмотри, как слишком сильная внутренняя динамика мешает прогнозу.', 'Случайная сеть превращает историю в набор признаков. Обучение подбирает только линейное сочетание этих признаков; рекуррентные веса не меняются.', 'Это учебная echo-state модель с заданным устойчивым масштабом. Для реальных данных нужна проверка памяти, спектрального радиуса и прогноза без подстановки истинного предыдущего значения.')

add('ae anomaly', 'Обучить сжатие и восстановление обычных объектов, затем сравнить ошибку на необычном объекте.', '''
normal = torch.randn(300, 1) @ torch.tensor([[1.,2.,-1.,0.5]])
normal += 0.03*torch.randn_like(normal)
model = nn.Sequential(nn.Linear(4, 1), nn.Linear(1, 4))
fit_regression(model, normal[:240], normal[:240], steps=300)
with torch.no_grad():
    ordinary = (model(normal[240:])-normal[240:]).square().mean(1)
    strange = torch.tensor([[5.,-5.,5.,-5.]])
    unusual = (model(strange)-strange).square().mean()
print('Обычные: средняя ошибка', ordinary.mean().item())
print('Необычный: ошибка', unusual.item())
''', 'На объекте, нарушающем обычную связь признаков, ошибка восстановления должна быть больше.', 'Добавь в обучение необычные объекты. Проверь, почему после этого они могут перестать выделяться по ошибке.', 'Узкое место из одного числа заставляет модель сохранить главную связь между четырьмя признаками. Объект с другой связью трудно восстановить.', 'Высокая ошибка — повод проверить объект, а не доказательство неисправности. Порог выбирают по отдельным данным; также возможны ложные тревоги и пропуски.')

add('vae', 'Обучить вероятностный автоэнкодер на точках и получить новые точки из скрытого пространства.', '''
x = torch.randn(256, 2)*0.3 + torch.tensor([1.,-1.])
encoder = nn.Linear(2, 4)
decoder = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=0.03)
for _ in range(200):
    mu, logvar = encoder(x).chunk(2, dim=1)
    logvar = logvar.clamp(-8, 8)
    z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
    reconstruction = F.mse_loss(decoder(z), x)
    kl = 0.5*(mu.square()+logvar.exp()-1-logvar).mean()
    loss = reconstruction + 0.1*kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    print('Новые точки:', decoder(torch.randn(5, 2)).tolist())
    print('Ошибка восстановления:', reconstruction.item())
''', 'Генератор выдаёт точки около области обучающих данных. Они не обязаны совпадать с конкретными строками учебного набора.', 'Сравни коэффициенты 0.001 и 1.0 перед kl. Что происходит с восстановлением и разнообразием?', 'Encoder выдаёт среднее и разброс распределения. Случайный шум создаёт скрытую точку, decoder восстанавливает данные. Дополнительная ошибка приближает скрытое распределение к стандартному нормальному.', 'Это линейный VAE на двух числах. Он не предназначен для генерации фотографий; качество генерации не измеряется одной ошибкой восстановления.')

add('gan', 'Понаблюдать за соревнованием генератора чисел и сети, отличающей их от настоящих примеров.', '''
generator = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
critic = nn.Sequential(nn.Linear(1, 12), nn.SiLU(), nn.Linear(12, 1))
g_opt = torch.optim.AdamW(generator.parameters(), lr=0.01)
d_opt = torch.optim.AdamW(critic.parameters(), lr=0.01)
for _ in range(300):
    real = torch.randn(64, 1)*0.4+2
    fake = generator(torch.randn(64, 1))
    d_loss = F.binary_cross_entropy_with_logits(critic(real), torch.ones_like(real))
    d_loss += F.binary_cross_entropy_with_logits(critic(fake.detach()), torch.zeros_like(fake))
    d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
    # Генератор получает градиент через критика, но не меняет его веса.
    critic.requires_grad_(False)
    g_loss = F.binary_cross_entropy_with_logits(critic(fake), torch.ones_like(fake))
    g_opt.zero_grad(set_to_none=True); g_loss.backward(); g_opt.step()
    critic.requires_grad_(True)
with torch.no_grad():
    generated = generator(torch.randn(1000, 1))
print('Среднее и разброс:', generated.mean().item(), generated.std().item())
''', 'Цель — среднее около 2 и стандартное отклонение около 0.4. За короткий запуск результат может колебаться; слишком малый разброс означает потерю разнообразия.', 'Сделай critic значительно шире генератора. Сравни, помогает ли более сильный соперник обучению.', 'Сначала критик учится отличать настоящие числа от сгенерированных. Затем генератор получает штраф, если критик распознал подделку. detach разрывает градиент только на шаге обучения критика.', 'GAN остаются полезны в отдельных задачах, но обучение может быть неустойчивым. Этот опыт показывает механизм, а не предлагает GAN как универсальный первый выбор для генерации.')

add('diffusion', 'Обучить модель убирать добавленный шум при нескольких уровнях зашумления.', '''
model = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(250):
    clean = torch.randn(128, 1)*0.3+2
    sigma = torch.rand(128, 1)*0.8+0.05
    noisy = clean + sigma*torch.randn_like(clean)
    loss = F.mse_loss(model(torch.cat([noisy, sigma], 1)), clean)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    clean = torch.randn(100, 1)*0.3+2
    sigma = torch.full_like(clean, 0.5)
    noisy = clean+sigma*torch.randn_like(clean)
    restored = model(torch.cat([noisy, sigma], 1))
print('До очистки:', F.mse_loss(noisy, clean).item())
print('После очистки:', F.mse_loss(restored, clean).item())
''', 'Средняя ошибка после очистки должна уменьшиться. Это проверка удаления шума на новых числах.', 'Убери sigma из входа модели. Проверь качество отдельно на слабом и сильном шуме.', 'Модель получает зашумлённое значение и силу шума, а правильный ответ — исходное значение. Случайные уровни шума учат её работать на разных этапах очистки.', 'Это обучение denoiser, одного компонента диффузии. Полного обратного процесса генерации здесь нет; один вызов очистки не равен генерации диффузионной моделью.', kind='Обучение отдельного компонента')

add('autoregression', 'Обучить сеть предсказывать следующий символ в короткой циклической последовательности.', '''
sequence = 'abcabcabcabcabcabc'
vocab = {c:i for i,c in enumerate('abc')}
tokens = torch.tensor([vocab[c] for c in sequence])
x, y = tokens[:-1], tokens[1:]
model = nn.Sequential(nn.Embedding(3, 8), nn.Linear(8, 3))
fit_classifier(model, x, y)
current, generated = torch.tensor([0]), 'a'
with torch.no_grad():
    for _ in range(12):
        current = model(current).argmax(1)
        generated += 'abc'[int(current)]
print('Продолжение:', generated)
''', 'Ожидается повторение abc. Каждый новый символ становится входом следующего шага.', 'Замени обучающий текст на abacabab. Почему знание только последнего символа перестаёт быть достаточным?', 'Embedding превращает номер символа в обучаемые числа. Сеть выбирает следующий символ по предыдущему, а генерация многократно повторяет этот шаг.', 'Это модель переходов между символами, а не современная языковая модель. Пример объясняет способ генерации; для сложного текста нужен контекст, токенизация и гораздо больше данных.')

add('flows', 'Обучить обратимое преобразование нормального шума в числа со средним 3 и разбросом 0.5.', '''
data = torch.randn(300)*0.5+3
shift = nn.Parameter(torch.tensor(0.))
log_scale = nn.Parameter(torch.tensor(0.))
optimizer = torch.optim.Adam([shift, log_scale], lr=0.05)
for _ in range(300):
    z = (data-shift)*torch.exp(-log_scale)
    # Минус логарифм плотности с поправкой на изменение масштаба.
    loss = (0.5*z.square()+log_scale).mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Сдвиг:', shift.item(), 'масштаб:', log_scale.exp().item())
z = torch.randn(5)
samples = shift + log_scale.exp()*z
recovered = (samples-shift)/log_scale.exp()
print('Обратимость:', torch.allclose(z, recovered, atol=1e-5))
''', 'Сдвиг приближается к 3, масштаб — к 0.5; обратное преобразование возвращает исходный шум.', 'Убери log_scale из ошибки и посмотри, почему модель стремится бесконечно увеличивать масштаб.', 'Однозначное преобразование позволяет и генерировать данные, и вычислять их плотность. Поправка log_scale учитывает, что растяжение распределяет вероятность по большему пространству.', 'Это одномерный affine flow. Для сложных распределений нужны композиции нелинейных обратимых слоёв.')

add('flowmatching lab-generative', 'Научить поле скоростей переносить облако шума к облаку данных.', '''
velocity = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 2))
optimizer = torch.optim.AdamW(velocity.parameters(), lr=0.01)
for _ in range(350):
    source = torch.randn(128, 2)
    target = torch.randn(128, 2)*0.3 + torch.tensor([2.,-1.])
    time = torch.rand(128, 1)
    position = (1-time)*source + time*target
    loss = F.mse_loss(velocity(torch.cat([position, time], 1)), target-source)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    samples = torch.randn(500, 2)
    for step in range(60):
        time = torch.full((500, 1), step/60)
        samples += velocity(torch.cat([samples, time], 1))/60
print('Среднее новых точек:', samples.mean(0).tolist())
print('Разброс:', samples.std(0).tolist())
''', 'Среднее сгенерированных точек приближается к [2, −1], разброс — к [0.3, 0.3]. Небольшие отклонения нормальны для такого короткого обучения.', 'Сравни 5, 20 и 100 шагов генерации при одной обученной модели. Что меняется в результате?', 'На обучении берём начало, конец и случайный момент между ними. Сеть предсказывает скорость движения в этой точке. На генерации остаётся только шум: сеть шаг за шагом переносит его в область данных.', 'Это полное маленькое обучение и генерация для двумерного облака. Качество картинок или больших генеративных моделей из него не следует.')

add('ebm', 'Найти вероятные состояния по обучаемой энергии, которая низкая около наблюдаемых данных.', '''
center = nn.Parameter(torch.tensor(0.))
optimizer = torch.optim.Adam([center], lr=0.05)
data = torch.randn(200)*0.2+1.5
for _ in range(100):
    loss = (data-center).square().mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
x = torch.randn(100)
for _ in range(80):
    x.requires_grad_(True)
    energy = 0.5*(x-center.detach()).square()
    gradient, = torch.autograd.grad(energy.sum(), x)
    x = (x-0.1*gradient+math.sqrt(0.02)*torch.randn_like(x)).detach()
print('Найденный центр:', center.item(), 'среднее образцов:', x.mean().item())
''', 'Образцы собираются около обученного центра 1.5. Шум не даёт всем образцам стать одной точкой.', 'Убери шум на шаге генерации. Чем поиск минимума энергии отличается от получения разнообразных образцов?', 'Энергия оценивает, насколько состояние не похоже на данные. Градиент указывает направление уменьшения энергии, а шум добавляет разнообразие.', 'Энергия здесь имеет заранее заданную квадратичную форму. Это не общая процедура обучения EBM: для произвольной энергии нужно учитывать нормировку или использовать специальные оценки градиента.', kind='Демонстрация механизма')

add('jepa', 'Обучить предсказатель восстанавливать скрытое представление другой части объекта.', '''
x = torch.randn(256, 4)
context = x[:, :2]
target_part = torch.stack([x[:,0]+x[:,1], x[:,0]-x[:,1]], 1)
# Фиксированный целевой encoder не может схлопнуться в одну константу.
target_encoder = nn.Linear(2, 8, bias=False).requires_grad_(False)
predictor = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 8))
with torch.no_grad():
    target = target_encoder(target_part)
fit_regression(predictor, context[:200], target[:200])
with torch.no_grad():
    print('Ошибка предсказания представления:', F.mse_loss(predictor(context[200:]), target[200:]).item())
''', 'Ошибка на новых объектах уменьшается: модель угадывает представление связанной части, а не сами пиксели.', 'Сделай target_part независимым случайным шумом. Объясни, почему его нельзя точно угадать по context.', 'Целевая сеть превращает одну часть объекта в признаки. Предсказатель по видимому контексту учится получать похожие признаки. Обучение идёт в пространстве представлений.', 'Это демонстрация предсказания представлений. В настоящих JEPA целевая сеть обычно тоже меняется по специальному правилу, а от схлопывания защищает вся схема обучения.', kind='Обучение отдельного компонента')

add('linear classification lab-table', 'По двум числовым признакам научиться предсказывать число и проверить ответ на новых строках.', '''
x = torch.randn(300, 2)
y = 3*x[:, :1] - 2*x[:, 1:] + 0.7 + 0.1*torch.randn(300, 1)
model = nn.Linear(2, 1)
fit_regression(model, x[:240], y[:240], steps=300)
with torch.no_grad():
    prediction = model(x[240:])
print('Ошибка на новых строках:', F.mse_loss(prediction, y[240:]).item())
print('Найденные коэффициенты:', model.weight.detach().tolist(), model.bias.item())
''', 'Коэффициенты приближаются к 3, −2 и 0.7. Ошибка на новых строках остаётся около уровня добавленного шума.', 'Добавь к ответу квадрат первого признака. Сравни линейный слой с маленькой сетью из темы MLP.', 'Каждый признак умножается на свой вес, затем результаты складываются со свободным сдвигом. Обучение находит веса, при которых предсказания близки к известным ответам.', 'Линейная модель — актуальная отправная точка для числовых данных. Она не описывает сложные взаимодействия без дополнительных признаков, но помогает понять, нужна ли более сложная сеть.')

add('trees', 'Подобрать первое правило дерева: по какому признаку и порогу разделить объекты.', '''
x, y = points(240)
best = None
for feature in range(2):
    for threshold in torch.linspace(-2, 2, 30):
        left = x[:180, feature] < threshold
        if not left.any() or left.all():
            continue
        a = y[:180][left].float().mean() >= 0.5
        b = y[:180][~left].float().mean() >= 0.5
        predicted = torch.where(left, a, b)
        error = (predicted != y[:180]).float().mean().item()
        if best is None or error < best[0]:
            best = (error, feature, threshold, a, b)
_, feature, threshold, a, b = best
predicted = torch.where(x[180:, feature] < threshold, a, b)
print('Признак:', feature, 'порог:', threshold.item())
print('Верно на новых строках:', (predicted == y[180:]).float().mean().item())
''', 'Получается одно понятное правило. На новых строках оно лучше случайного выбора, но уступает границе, учитывающей оба признака.', 'После первого разбиения подбери отдельное правило внутри каждой половины. Следи, не становятся ли листья слишком маленькими.', 'Дерево перебирает вопросы вида «число меньше порога?» и выбирает вопрос, после которого ответы в каждой группе более однородны.', 'Это дерево глубины 1. Современные ансамбли обучают множество деревьев с регуляризацией; для рабочих таблиц используй поддерживаемую библиотеку, а этот код — для понимания разбиения.')

add('svm', 'Обучить разделяющую прямую, которая оставляет запас до ближайших объектов.', '''
x, labels = points(240)
y = labels.float()*2-1
model = nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
for _ in range(250):
    score = model(x[:180]).squeeze(1)
    loss = (1-y[:180]*score).clamp_min(0).mean()+0.01*model.weight.square().sum()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = (model(x[180:]).squeeze(1)>0).long()
print('Верно:', prediction.eq(labels[180:]).float().mean().item())
''', 'Модель разделяет точки с высокой долей верных ответов. Объекты слишком близко к границе штрафуются, даже если сторона выбрана правильно.', 'Увеличь коэффициент 0.01 перед штрафом весов. Сравни размер весов и качество.', 'Правильные ответы переведены в −1 и +1. Произведение ответа на оценку модели должно быть больше 1 — это запас уверенности до границы.', 'Это линейный soft-margin принцип с hinge loss, без ядрового преобразования. Для больших современных задач выбирай модель по качеству и ресурсам, а не по возрасту названия.')

add('knn', 'Классифицировать новую точку по трём ближайшим размеченным соседям.', '''
x, y = points(240)
distance = torch.cdist(x[180:], x[:180])
neighbors = distance.topk(3, largest=False).indices
prediction = y[:180][neighbors].float().mean(1).ge(0.5).long()
print('Верно на новых точках:', prediction.eq(y[180:]).float().mean().item())
print('Соседи первой новой точки:', neighbors[0].tolist())
''', 'Большинство точек получают категорию своих соседей. Веса модели не обучались: метод хранит примеры.', 'Умножь один признак на 100. Почему без нормализации расстояние перестаёт отражать полезное сходство?', 'cdist считает расстояния от новых объектов до учебных. topk выбирает ближайшие; большинство голосов задаёт ответ.', 'Полный перебор дорог на больших коллекциях. Рабочий поиск использует индексы приближённых соседей и осмысленные признаки; здесь показана точная проверка на небольшом наборе.')

add('naivebayes', 'Оценить категорию по распределению каждого отдельного признака.', '''
x = torch.cat([torch.randn(120, 2)-1, torch.randn(120, 2)+1])
y = torch.cat([torch.zeros(120), torch.ones(120)]).long()
order = torch.randperm(len(x)); x, y = x[order], y[order]
means, variances, priors = [], [], []
for label in range(2):
    subset = x[:180][y[:180] == label]
    means.append(subset.mean(0)); variances.append(subset.var(0)+1e-4)
    priors.append(len(subset)/180)
mu, var = torch.stack(means), torch.stack(variances)
log_score = -0.5*((x[180:, None]-mu).square()/var+var.log()).sum(2)
log_score += torch.tensor(priors).log()
print('Верно:', log_score.argmax(1).eq(y[180:]).float().mean().item())
''', 'Для двух разнесённых облаков точек получается высокая доля правильных ответов без градиентного обучения.', 'Добавь почти полную копию первого признака. Объясни, почему модель может стать чрезмерно уверенной.', 'Для каждого класса вычисляются среднее и разброс признаков. Вероятности отдельных признаков перемножаются; логарифмы превращают произведение в сумму.', 'Предположение независимости признаков часто неверно. Значения scores ещё не являются проверенными вероятностями: уверенность нужно отдельно калибровать.')

add('cluster unsupervised', 'Разделить точки на группы без заранее заданных названий категорий.', '''
x = torch.cat([torch.randn(100, 2)*0.3-1, torch.randn(100, 2)*0.3+1])
centers = x[torch.randperm(len(x))[:2]].clone()
for _ in range(20):
    labels = torch.cdist(x, centers).argmin(1)
    for k in range(2):
        if (labels == k).any():
            centers[k] = x[labels == k].mean(0)
print('Центры групп:', centers.tolist())
print('Число точек в группах:', torch.bincount(labels, minlength=2).tolist())
''', 'Центры должны оказаться около [−1, −1] и [1, 1]. Номера групп произвольны и могут поменяться местами.', 'Задай три центра вместо двух. Почему алгоритм всё равно создаёт три группы, даже если естественных облаков два?', 'Точки относятся к ближайшему центру, после чего центры перемещаются в среднее своих точек. Шаги повторяются до стабилизации.', 'Это k-means: он ищет компактные группы по расстоянию, а не смысловые категории. Число групп и масштаб признаков задаёт человек.')

add('decomposition', 'Сжать почти двумерные данные до одной главной координаты и измерить потерю.', '''
t = torch.linspace(-2, 2, 120)
x = torch.stack([t, 2*t, -t], dim=1)+0.05*torch.randn(120, 3)
mean = x[:90].mean(0)
u, s, vh = torch.linalg.svd(x[:90]-mean, full_matrices=False)
direction = vh[:1]
compressed = (x[90:]-mean) @ direction.T
restored = compressed @ direction + mean
print('Размер до и после:', tuple(x[90:].shape), tuple(compressed.shape))
print('Ошибка восстановления:', F.mse_loss(restored, x[90:]).item())
''', 'Три числа на объект заменяются одним. Ошибка мала, потому что признаки в основном описывают одну и ту же величину.', 'Добавь независимый сильный шум ко второму признаку. Как меняется направление главной компоненты?', 'Главная компонента — направление наибольшего разброса учебных данных. Проекция на это направление сохраняет большую часть изменчивости.', 'PCA сохраняет разброс, а не обязательно полезность для предсказания. Центрирование и компоненты вычислены только по учебной части, чтобы не подсматривать в проверочную.')

add('isolation', 'Сравнить, сколько случайных разрезов требуется, чтобы изолировать обычную и далёкую точку.', '''
x = torch.cat([torch.randn(100, 2)*0.3, torch.tensor([[5.,5.]])])
def depth(values, index, level=0):
    if len(values)<=1 or level>=30:
        return level
    feature = int(torch.randint(0, values.shape[1], (1,)))
    low, high = values[:, feature].min(), values[:, feature].max()
    if high == low:
        return level
    cut = low + torch.rand(())*(high-low)
    side = values[:, feature] < cut
    keep = side == side[index]
    new_index = int(keep[:index].sum())
    return depth(values[keep], new_index, level+1)
for index in [0, 100]:
    paths = [depth(x, index) for _ in range(100)]
    print('Точка', index, 'средняя глубина:', sum(paths)/len(paths))
''', 'Далёкая точка обычно отделяется меньшим числом разрезов. Её средний путь короче.', 'Поставь рядом с далёкой точкой ещё 20 похожих. Посмотри, почему аномальность зависит от окружения.', 'Случайные разрезы быстро отделяют редкие объекты в пустых областях. Плотные группы приходится разделять дольше.', 'Это демонстрация длины пути. Полный Isolation Forest использует подвыборки, множество деревьев и нормировку глубины; здесь не реализован готовый производственный детектор.', kind='Демонстрация механизма')

add('timeseries classicaltime', 'Предсказать следующий отсчёт по предыдущим шести, проверяя только более поздние наблюдения.', '''
signal = torch.sin(torch.arange(300)*0.1)+0.05*torch.randn(300)
windows = signal.unfold(0, 7, 1)
x, y = windows[:, :6], windows[:, 6:]
model = nn.Linear(6, 1)
fit_regression(model, x[:220], y[:220])
with torch.no_grad():
    prediction = model(x[220:])
print('Ошибка модели:', F.mse_loss(prediction, y[220:]).item())
print('Ошибка «повтори последнее»:', F.mse_loss(x[220:, -1:], y[220:]).item())
''', 'Сравниваются модель и простой прогноз последним значением. На периодическом сигнале модель обычно точнее.', 'Измени частоту колебаний только после позиции 220. Проверь, как смена режима влияет на прогноз.', 'Каждая строка содержит шесть последовательных отсчётов, ответ — следующий. Проверочные окна идут после учебных, чтобы будущее не попало в прошлое.', 'Это линейный авторегрессионный прогноз, а не полноценная ARIMA. Проверка использует настоящую предыдущую историю; для многодневного прогноза отдельно проверяют накопление ошибок.')

add('bayes probability', 'Обновить мнение о вероятности успеха после нескольких наблюдений.', '''
# До опыта допускаем все вероятности равномерно: Beta(1, 1).
a, b = 1., 1.
observations = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])
a += observations.sum().item()
b += (1-observations).sum().item()
posterior = torch.distributions.Beta(a, b)
samples = posterior.sample((10000,))
interval = torch.quantile(samples, torch.tensor([0.025, 0.975]))
print('Оценка вероятности:', a/(a+b))
print('95% интервал:', interval.tolist())
''', 'После шести успехов и двух неудач средняя оценка равна 0.7. Интервал остаётся широким: наблюдений мало.', 'Повтори те же доли успехов на 800 наблюдениях. Сравни ширину интервала, а не только среднее.', 'До данных задаётся распределение возможных вероятностей. Каждый успех увеличивает a, каждая неудача — b. Получается новое распределение, учитывающее и данные, и начальное мнение.', 'Интервал относится к вероятности в выбранной модели независимых испытаний. Если условия меняются, такой расчёт может быть слишком уверенным.')

add('gp', 'Получить прогноз функции и разброс там, где наблюдений мало.', '''
x = torch.tensor([[-1.], [0.], [1.]])
y = x.sin()
query = torch.linspace(-3, 3, 21)[:, None]
def kernel(a, b):
    return torch.exp(-0.5*torch.cdist(a, b).square())
k = kernel(x, x)+0.01*torch.eye(len(x))
cross = kernel(query, x)
mean = cross @ torch.linalg.solve(k, y)
variance = (1-(cross*torch.linalg.solve(k, cross.T).T).sum(1)).clamp_min(0)
for i in [0, 10, 20]:
    print('x =', query[i].item(), 'среднее =', mean[i].item(), 'разброс =', variance[i].sqrt().item())
''', 'Около наблюдаемого нуля разброс меньше, чем у краёв −3 и 3, где измерений нет.', 'Уменьши длину связи: в kernel дели расстояния на 0.3. Как быстро теряется влияние соседнего измерения?', 'Ядро задаёт предположение: близкие точки должны иметь похожие ответы. Формулы условного нормального распределения дают и среднее, и неопределённость.', 'Это маленькая точная GP-регрессия с фиксированным ядром. Кубическая стоимость решения системы ограничивает число наблюдений; крупным задачам нужны приближения.')

add('states', 'Восстановить плавное скрытое значение из шумных измерений.', '''
truth = torch.linspace(0, 2, 60)
observed = truth+0.35*torch.randn(60)
mean, variance, estimates = torch.tensor(0.), torch.tensor(1.), []
for measurement in observed:
    variance = variance+0.01  # Допускаем изменение состояния между шагами.
    gain = variance/(variance+0.35**2)
    mean = mean+gain*(measurement-mean)
    variance = (1-gain)*variance
    estimates.append(mean)
estimates = torch.stack(estimates)
print('Ошибка измерений:', F.mse_loss(observed, truth).item())
print('Ошибка фильтра:', F.mse_loss(estimates, truth).item())
''', 'Фильтр обычно уменьшает шум, но может отставать от быстро меняющегося состояния.', 'Сделай резкий скачок truth в середине. Увеличь 0.01 и посмотри на компромисс между быстротой и шумом.', 'Фильтр хранит оценку и её неопределённость. При точном измерении доверяет наблюдению сильнее; при шумном — сильнее опирается на прошлую оценку.', 'Это одномерный фильтр Калмана со случайным блужданием. Он показывает непрерывные скрытые состояния, а не весь класс скрытых марковских моделей.')

add('mixtures', 'Представить данные как смесь двух распределений и оценить, к какой части относится каждая точка.', '''
x = torch.cat([torch.randn(120)*0.3-1.5, torch.randn(120)*0.3+1.5])
means = torch.tensor([-1., 1.]); variance = torch.ones(2); weights = torch.ones(2)/2
for _ in range(30):
    score = -0.5*((x[:, None]-means).square()/variance+variance.log())+weights.log()
    responsibility = score.softmax(1)
    count = responsibility.sum(0)
    means = (responsibility*x[:, None]).sum(0)/count
    variance = (responsibility*(x[:, None]-means).square()).sum(0)/count
    variance = variance.clamp_min(1e-4)
    weights = count/len(x)
print('Центры:', means.tolist(), 'доли:', weights.tolist())
''', 'Центры сходятся примерно к −1.5 и +1.5, доли — к половине данных на каждую часть.', 'Положи обе начальные means в одну точку. Почему симметричная инициализация может помешать разделению?', 'Сначала каждому объекту назначаются мягкие вероятности принадлежности. Затем центры и разбросы пересчитываются с этими весами; два шага повторяются.', 'Это одномерная гауссова смесь с EM. Число частей задано заранее; разные запуски могут находить разные решения.')

add('causal', 'Увидеть разницу между наблюдаемой связью и причинным эффектом вмешательства.', '''
weather = torch.randn(10000)
ice_cream = weather+0.2*torch.randn(10000)
swimming = 2*weather+0.2*torch.randn(10000)
correlation = torch.corrcoef(torch.stack([ice_cream, swimming]))[0,1]
# В нашей причинной схеме купание зависит от погоды, не от мороженого.
intervention = torch.full_like(ice_cream, 100.)
swimming_after = 2*weather+0.2*torch.randn(10000)
print('Корреляция:', correlation.item())
print('Изменение среднего купания при вмешательстве:', (swimming_after-swimming).mean().item())
assert intervention.mean() > ice_cream.mean()
''', 'Продажи мороженого и купание сильно связаны. Принудительная смена продаж почти не меняет купание, потому что общая причина — погода.', 'Добавь в уравнение swimming член 0.3*ice_cream и пересчитай результат после intervention. Теперь возникает причинный эффект.', 'Обычная статистика видит совместное изменение двух величин. Причинная модель описывает, какое уравнение меняется при вмешательстве.', 'Причинная схема здесь известна, потому что мы сами создали данные. В реальной задаче она требует обоснования; высокая корреляция и код сами по себе её не устанавливают.', kind='Демонстрация причинной модели')

add('text lab-text', 'Обучить компактный классификатор двух команд по словам и проверить новые формулировки.', '''
texts = ['включи свет', 'свет включить', 'зажги лампу', 'лампу включить',
         'выключи свет', 'свет выключить', 'погаси лампу', 'лампу выключить']
labels = torch.tensor([1,1,1,1,0,0,0,0])
vocab = {word:i for i,word in enumerate(sorted(set(' '.join(texts).split())))}
def encode(sentences):
    result = torch.zeros(len(sentences), len(vocab))
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            if word in vocab:
                result[i, vocab[word]] += 1
    return result
model = nn.Linear(len(vocab), 2)
fit_classifier(model, encode(texts), labels)
checks = ['включи лампу', 'выключи лампу', 'пожалуйста свет включить']
with torch.no_grad():
    prediction = model(encode(checks)).argmax(1)
for sentence, answer in zip(checks, prediction):
    print(sentence, '→', 'включить' if answer else 'выключить')
''', 'Новые сочетания знакомых слов должны получить правильное действие. Незнакомое «пожалуйста» будет проигнорировано.', 'Проверь «не включай свет». Почему подсчёт слов может ошибиться? Добавь отрицания в данные и сравни с моделью, учитывающей порядок.', 'Словарь создаётся только по учебным фразам. Каждая фраза превращается в числа — сколько раз встретилось слово. Линейный слой учится связывать эти числа с действием.', 'Это маленький измеримый базовый вариант для фиксированного набора команд. Он не понимает язык в общем смысле и не извлекает даты. Для сложных формулировок нужен современный предобученный encoder или языковая модель.')

add('recommend', 'По известным оценкам научить маленькие векторы пользователей и предметов предсказывать совместимость.', '''
true_users, true_items = torch.randn(12, 3), torch.randn(10, 3)
rating = true_users @ true_items.T
users, items = nn.Embedding(12, 3), nn.Embedding(10, 3)
mask = torch.rand(12, 10) < 0.8
optimizer = torch.optim.AdamW(list(users.parameters())+list(items.parameters()), lr=0.04)
for _ in range(400):
    predictions = users.weight @ items.weight.T
    loss = F.mse_loss(predictions[mask], rating[mask])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = users.weight @ items.weight.T
print('Ошибка на скрытых оценках:', F.mse_loss(prediction[~mask], rating[~mask]).item())
''', 'Ошибка на скрытых сочетаниях показывает, удалось ли восстановить предпочтения, а не только известные оценки.', 'Уменьши долю известных оценок с 0.8 до 0.3. Проверь пользователей, у которых почти нет истории.', 'Пользователь и предмет описываются короткими обучаемыми векторами. Их скалярное произведение даёт оценку совместимости; обучение приближает её к наблюдениям.', 'Это матричная факторизация на искусственных полных предпочтениях. В реальном сервисе отсутствие клика не равно отрицательной оценке; нужны временная проверка, кандидаты и обработка новых пользователей.')

add('multimodal', 'Обучить модель соединять две неполные подсказки от разных источников.', '''
image_feature = torch.randn(400, 2)
text_feature = torch.randn(400, 2)
y = (image_feature[:,0]+text_feature[:,0]>0).long()
x = torch.cat([image_feature, text_feature], 1)
model = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
fit_classifier(model, x[:320], y[:320])
print('Оба источника:', accuracy(model, x[320:], y[320:]))
missing = x[320:].clone(); missing[:, 2:] = 0
print('Без второго источника:', accuracy(model, missing, y[320:]))
''', 'С двумя источниками качество обычно выше: правильный ответ зависит от информации в обоих.', 'Обучи сеть иногда получать нули вместо второго источника. Станет ли она устойчивее к его отсутствию?', 'Один источник даёт часть признаков, другой — другую часть. Сеть получает объединённый вектор и учится учитывать их вместе.', 'Числа имитируют уже извлечённые признаки; здесь не обучается распознавание настоящих изображений и текста. Это опыт со слиянием источников, не готовая мультимодальная модель.')

add('controltask imitation', 'Научить сеть повторять простое управление движением к нулю.', '''
x = torch.randn(400, 2)  # положение и скорость
expert = (-0.7*x[:, :1]-0.3*x[:, 1:]).clamp(-1, 1)
model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Tanh())
fit_regression(model, x[:320], expert[:320])
with torch.no_grad():
    print('Ошибка повторения эксперта:', F.mse_loss(model(x[320:]), expert[320:]).item())
    position, speed = 2., 0.
    for _ in range(60):
        action = float(model(torch.tensor([[position, speed]])))
        speed += 0.1*action
        position += 0.1*speed
print('Положение после управления:', position)
''', 'Сеть приближает действия эксперта на новых состояниях. В короткой симуляции положение приближается к цели, но может её пересечь.', 'Начни симуляцию из положения 10. Объясни, почему хорошее копирование на учебном диапазоне не гарантирует управление далеко за его пределами.', 'Правильные действия задаёт известное правило. Сеть учится повторять их, затем её ответ используется внутри цикла движения.', 'Это имитация в простой симуляции. Маленькая ошибка действий не доказывает устойчивость управления; на реальное оборудование этот код переносить нельзя без отдельной модели, ограничений и проверки.')

add('pinn lab-science', 'Обучить сеть находить функцию по уравнению u′(x) = −u(x) и условию u(0) = 1.', '''
model = nn.Sequential(nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(400):
    x = torch.rand(80, 1, requires_grad=True)*2
    u = model(x)
    derivative, = torch.autograd.grad(u.sum(), x, create_graph=True)
    equation_error = (derivative+u).square().mean()
    boundary_error = (model(torch.zeros(1,1))-1).square().mean()
    loss = equation_error+boundary_error
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    x = torch.linspace(0, 2, 50)[:, None]
    print('Ошибка относительно exp(−x):', F.mse_loss(model(x), torch.exp(-x)).item())
''', 'Ответ приближается к exp(−x). Правильную функцию использовали только для проверки, не для обучения.', 'Убери boundary_error. Почему уравнение само по себе допускает больше одного решения?', 'Автоматическое дифференцирование вычисляет производную выхода сети по входу. Обучение штрафует нарушение уравнения и начального условия.', 'Здесь простое устойчивое уравнение в одном измерении. PINN не всегда точнее или быстрее численных решателей; сложные уравнения требуют отдельного анализа ошибок.')

add('neuraloperator', 'Обучить преобразование целого сигнала в сглаженный сигнал через частоты.', '''
x = torch.randn(180, 32)
freq = torch.fft.rfft(x)
response = torch.exp(-torch.arange(freq.shape[1]).float()/4)
y = torch.fft.irfft(freq*response, n=32)
weights = nn.Parameter(torch.ones(freq.shape[1], dtype=torch.complex64))
optimizer = torch.optim.Adam([weights], lr=0.05)
for _ in range(180):
    prediction = torch.fft.irfft(torch.fft.rfft(x[:140])*weights, n=32)
    loss = F.mse_loss(prediction, y[:140])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    prediction = torch.fft.irfft(torch.fft.rfft(x[140:])*weights, n=32)
print('Ошибка на новых сигналах:', F.mse_loss(prediction, y[140:]).item())
''', 'Модель учится сохранять медленные изменения и подавлять быстрые, получая целую выходную последовательность.', 'Обучи на другом затухании частот. Затем проверь, что будет с сигналом, которого не было в обучении.', 'Преобразование Фурье переводит сигнал в частоты. Обучаемые коэффициенты меняют каждую частоту, обратное преобразование возвращает сигнал.', 'Это обучаемый линейный спектральный оператор. Полноценные FNO добавляют нелинейности, каналы и работу с разными функциями; перенос на новую сетку этим опытом не проверяется.')

add('symbolic hybrids', 'Соединить вероятностный ответ сети с явным правилом допустимости.', '''
# Модель предлагает действия: читать, отправить, удалить.
logits = torch.tensor([[1., 2., 5.]])
allowed = torch.tensor([[True, True, False]])
restricted = logits.masked_fill(~allowed, float('-inf'))
probabilities = restricted.softmax(1)
action = probabilities.argmax(1).item()
print('Вероятности допустимых действий:', probabilities.tolist())
print('Выбрано действие:', ['читать', 'отправить', 'удалить'][action])
assert probabilities[0, 2] == 0
''', 'Самая высокая исходная оценка была у удаления, но правило запрещает его. Выбирается отправка.', 'Запрети все действия. Добавь явную проверку allowed.any(), которая вместо softmax возвращает отказ.', 'Нейросеть оценивает подходящие варианты, а правило отсеивает недопустимые. Ограничение проверяется в отдельном шаге, а не надеется на послушание модели.', 'Это пример маски допустимых действий, а не общая система логического вывода и не полноценная защита агента. Правила должны применяться там, где выполняется действие.', kind='Демонстрация механизма')

add('symbolicreg', 'Подобрать понятную формулу, сравнив несколько простых выражений.', '''
x = torch.linspace(-2, 2, 100)
y = 2*x.square()+0.5
candidates = {
    'a*x+b': lambda t: torch.stack([t, torch.ones_like(t)], 1),
    'a*x²+b': lambda t: torch.stack([t.square(), torch.ones_like(t)], 1),
    'a*sin(x)+b': lambda t: torch.stack([t.sin(), torch.ones_like(t)], 1),
}
for name, features in candidates.items():
    weights = torch.linalg.lstsq(features(x[:70]), y[:70]).solution
    error = F.mse_loss(features(x[70:])@weights, y[70:])
    print(name, 'коэффициенты', weights.tolist(), 'ошибка', error.item())
''', 'Квадратичная формула восстанавливает коэффициенты 2 и 0.5 с практически нулевой ошибкой.', 'Добавь шум и более сложные кандидаты. Выбирай формулу по новым точкам и штрафуй лишнюю сложность.', 'Перебирается не только набор коэффициентов, но и вид выражения. Это позволяет получить ответ, который можно прочитать как формулу.', 'Это перебор трёх кандидатов, не полноценный поиск выражений. Современная символьная регрессия вроде PySR исследует гораздо больше структур и контролирует сложность.')

add('evolution', 'Найти два параметра без производных: сохранять лучшие решения и слегка изменять их.', '''
population = torch.randn(100, 2)*3
target = torch.tensor([1.5, -0.7])
for _ in range(50):
    score = (population-target).square().sum(1)
    elite = population[score.topk(10, largest=False).indices]
    children = elite.repeat(9, 1)+0.08*torch.randn(90, 2)
    population = torch.cat([elite, children])
best = population[((population-target).square().sum(1)).argmin()]
print('Найденные параметры:', best.tolist())
''', 'Параметры приближаются к [1.5, −0.7]. Старые лучшие решения сохраняются, пока потомки исследуют окрестность.', 'Постепенно уменьшай масштаб мутаций с 0.5 до 0.01. Сравни скорость поиска и точность.', 'Вместо градиента используются оценка результата, отбор и случайные изменения. Это применимо, когда оценка недифференцируема или доступна только как чёрный ящик.', 'На гладкой задаче градиентный метод обычно гораздо экономнее. Эволюция здесь демонстрирует поиск, а не преимущество для обучения больших нейросетей.')

add('snn', 'Проследить накопление напряжения и появление импульсов в модели нейрона.', '''
current = torch.tensor([0.4]*20)
voltage, spikes = torch.tensor(0.), []
for value in current:
    voltage = 0.9*voltage+value
    spike = (voltage>=1).float()
    voltage = voltage-spike
    spikes.append(int(spike))
print('Импульсы:', spikes)
print('Всего импульсов:', sum(spikes))
''', 'Импульс возникает не на каждом шаге: сначала накапливается напряжение. После импульса оно уменьшается.', 'Подай ток 0.2, затем 0.8. Сравни частоту импульсов и эффект коэффициента утечки 0.9.', 'Нейрон суммирует поступающий ток, часть накопленного сигнала теряется, а достижение порога создаёт короткий импульс.', 'Это прямой расчёт нейрона LIF без обучения. Порог недифференцируем; обучение SNN обычно требует surrogate gradient или другого специального метода. Экономия энергии зависит от оборудования.', kind='Демонстрация механизма')

add('quantum', 'Обучить угол вращения одного кубита в классической симуляции.', '''
angle = nn.Parameter(torch.tensor(0.2))
optimizer = torch.optim.Adam([angle], lr=0.05)
# После RY(theta) вероятности двух исходов: cos²(theta/2), sin²(theta/2).
for _ in range(150):
    probability_one = torch.sin(angle/2).square()
    loss = (probability_one-0.8).square()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Угол:', angle.item())
print('Вероятность исхода 1:', torch.sin(angle/2).square().item())
''', 'Вероятность исхода 1 приближается к 0.8; обучение меняет угол параметризованной операции.', 'Добавь симуляцию 100 измерений через torch.bernoulli. Сравни измеренную долю с точной вероятностью.', 'Состояние кубита задаёт вероятности исходов. Параметризованное вращение меняет их, а классический оптимизатор подбирает угол по ошибке.', 'Это симуляция одного кубита средствами PyTorch, не квантовое устройство и не доказательство квантового преимущества. QNN остаются исследовательским направлением, а не универсальным способом сделать маленькую модель умнее.', kind='Исследовательская демонстрация')

add('hdc', 'Закодировать объекты длинными случайными векторами и классифицировать по прототипам.', '''
x, y = points(300)
projection = torch.randn(2, 1024)
codes = torch.where(x@projection>=0, 1., -1.)
prototypes = torch.stack([codes[:240][y[:240]==k].sum(0) for k in range(2)])
prototypes = F.normalize(prototypes, dim=1)
prediction = (F.normalize(codes[240:], dim=1)@prototypes.T).argmax(1)
print('Доля верных ответов:', prediction.eq(y[240:]).float().mean().item())
''', 'Категория определяется сходством с суммарным прототипом. Градиентного обучения в этом опыте нет.', 'Сравни длину кода 32, 256 и 4096. Есть ли смысл увеличивать размер после стабилизации качества?', 'Каждому объекту соответствует длинный код из −1 и +1. Коды учебных объектов одной категории складываются в прототип, с которым сравнивают новые объекты.', 'Случайные коды не создают смысл сами по себе. Здесь HDC демонстрируется на простой геометрии; качество и экономия памяти зависят от способа кодирования и аппаратной упаковки битов.')

add('supervised scratch', 'Пройти весь цикл обучения с нуля: данные, ошибка, обновление весов, проверка.', '''
x, y = points()
model = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
print('До обучения:', accuracy(model, x[240:], y[240:]))
loss = fit_classifier(model, x[:240], y[:240])
print('После обучения:', accuracy(model, x[240:], y[240:]))
print('Последняя ошибка на учебной части:', loss)
''', 'На новых объектах сеть отвечает лучше после обучения. Учебная ошибка и доля верных ответов — разные числа.', 'Обучи на 10 объектах вместо 240 и повтори с несколькими seed. Насколько надёжен вывод по одному запуску?', 'Сеть начинает со случайными весами. Для каждого примера есть правильная категория; ошибка сравнивает предсказание с ней, а оптимизатор корректирует веса.')

add('selfsupervised masked', 'Обучить сеть восстанавливать скрытые элементы, не создавая ручные метки.', '''
t = torch.randn(320, 1)
x = torch.cat([t, 2*t, -t, 0.5*t], 1)+0.03*torch.randn(320, 4)
model = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 4))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(220):
    mask = torch.rand(250, 4)<0.35
    observed = x[:250].masked_fill(mask, 0)
    prediction = model(torch.cat([observed, mask.float()], 1))
    loss = ((prediction-x[:250]).square()*mask).sum()/mask.sum().clamp_min(1)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    mask = torch.zeros(70, 4, dtype=torch.bool); mask[:, 1] = True
    prediction = model(torch.cat([x[250:].masked_fill(mask, 0), mask.float()], 1))
print('Ошибка на скрытом признаке новых объектов:', F.mse_loss(prediction[:,1], x[250:,1]).item())
''', 'Сеть восстанавливает второй признак по связанным видимым признакам. Человек не размечал правильные ответы: они были в исходных данных.', 'Сделай все четыре признака независимыми. Почему сеть уже не может точно восстановить скрытый?', 'Во время обучения случайно прячется часть данных, а ошибка считается только на спрятанном. Маска сообщает модели, где настоящее нулевое значение, а где пропуск.', 'Это masked reconstruction. Успех в восстановлении не гарантирует полезных признаков для любой другой задачи; это нужно отдельно проверять.')

add('semisupervised', 'Использовать уверенные предсказания учителя как временные метки для неразмеченных объектов.', '''
x, y = points(500)
model = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 2))
fit_classifier(model, x[:30], y[:30])
print('Только 30 меток:', accuracy(model, x[400:], y[400:]))
with torch.no_grad():
    confidence, pseudo = model(x[30:400]).softmax(1).max(1)
    keep = confidence>0.95
augmented_x = torch.cat([x[:30], x[30:400][keep]])
augmented_y = torch.cat([y[:30], pseudo[keep]])
fit_classifier(model, augmented_x, augmented_y)
print('Добавлено временных меток:', int(keep.sum()))
print('После дообучения:', accuracy(model, x[400:], y[400:]))
''', 'Код показывает число добавленных меток и качество до и после. Улучшение не гарантировано: уверенная ошибка тоже может попасть в обучение.', 'Сравни пороги 0.6 и 0.99. Запиши, как меняются число псевдометок и качество на проверочной части.', 'Сначала модель учится на маленькой размеченной части. Затем её самые уверенные ответы на остальных примерах добавляются как псевдометки. Проверочные метки остаются только для оценки.')

add('transfer domain', 'Перенести обученное представление на немного изменившуюся задачу, обучая только новую голову.', '''
x, y = points(400)
encoder = nn.Sequential(nn.Linear(2, 16), nn.SiLU())
source = nn.Sequential(encoder, nn.Linear(16, 2))
fit_classifier(source, x[:300], y[:300])
encoder.requires_grad_(False)
target = nn.Sequential(encoder, nn.Linear(16, 2))
new_labels = (x[:,0]-0.3*x[:,1]>0.2).long()
fit_classifier(target, x[:60], new_labels[:60])
print('Качество новой головы:', accuracy(target, x[300:], new_labels[300:]))
print('Обучаемых параметров:', sum(p.numel() for p in target.parameters() if p.requires_grad))
''', 'Обучаются только веса новой головы. Качество показывает, насколько признаки исходной задачи подходят новой.', 'Измени задачу на распознавание круга. Сравни замороженный encoder и разрешённое дообучение всех слоёв.', 'Первая сеть уже научилась преобразовывать данные в признаки. Эти веса замораживаются, а новая небольшая часть учится читать признаки для другого ответа.', 'Предобучение здесь создаётся внутри опыта на маленькой задаче. В рабочем проекте используют подходящую актуальную предобученную модель, проверяя её лицензию, размер и качество на своих данных.')

add('lora', 'Адаптировать замороженное преобразование через две небольшие обучаемые матрицы.', '''
x = torch.randn(300, 8)
base = torch.randn(8, 8)*0.2
true_update = torch.randn(8, 2)@torch.randn(2, 8)*0.1
y = x@(base+true_update)
a = nn.Parameter(torch.randn(8, 2)*0.01)
b = nn.Parameter(torch.zeros(2, 8))
optimizer = torch.optim.AdamW([a, b], lr=0.04)
for _ in range(250):
    prediction = x[:240]@base + x[:240]@a@b
    loss = F.mse_loss(prediction, y[:240])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    print('Ошибка на новых объектах:', F.mse_loss(x[240:]@base+x[240:]@a@b, y[240:]).item())
print('Весов основной матрицы:', base.numel(), 'обучаемых:', a.numel()+b.numel())
''', 'Две матрицы содержат 32 обучаемых числа вместо 64. Они восстанавливают добавку ранга 2.', 'Поставь ранг 1, затем 4. Сравни ошибку и число обучаемых параметров.', 'Базовая матрица не меняется. Дополнение представлено произведением двух узких матриц, поэтому на обучение требуется меньше параметров и состояния оптимизатора.', 'Это точная идея низкоранговой добавки, но не дообучение готовой LLM. В настоящей PEFT-схеме важны масштаб адаптера, выбранные слои, токенизация и качество исходной модели.')

add('meta', 'Подготовить общую стартовую точку, из которой модель быстро подстраивается под новую прямую.', '''
initial = nn.Linear(1, 1)
for _ in range(120):
    task = copy.deepcopy(initial)
    x = torch.randn(12, 1)
    slope = 1+torch.rand(()); bias = torch.rand(())*0.4
    fit_regression(task, x, slope*x+bias, steps=10)
    with torch.no_grad():
        for shared, adapted in zip(initial.parameters(), task.parameters()):
            shared.lerp_(adapted, 0.15)
x = torch.randn(30, 1); y = 1.7*x+0.15
adapted = copy.deepcopy(initial)
fit_regression(adapted, x[:5], y[:5], steps=10)
with torch.no_grad():
    print('До адаптации:', F.mse_loss(initial(x[5:]), y[5:]).item())
    print('После 10 шагов:', F.mse_loss(adapted(x[5:]), y[5:]).item())
''', 'После нескольких шагов на новой задаче ошибка обычно уменьшается. Для вывода о пользе метаобучения нужно сравнение с обычной инициализацией на многих задачах.', 'Добавь контрольную сеть со случайной инициализацией и одинаковым числом шагов. Повтори опыт для 20 новых прямых.', 'Общая модель временно адаптируется к разным задачам. После каждой задачи её стартовые веса чуть сдвигаются к адаптированным — это упрощённая схема Reptile.', 'Это небольшой опыт с задачами из одного семейства. Он не доказывает способность быстро учиться чему угодно.')

add('icl', 'Обучить сеть читать примеры задачи из входа и отвечать без обновления весов на проверке.', '''
# В каждой задаче своё правило y=a*x+b. Контекст: две пары (x,y) и новый x.
def tasks(count):
    a, b = torch.rand(count, 1)*2-1, torch.rand(count, 1)*2-1
    x1, x2 = -torch.ones(count, 1), torch.ones(count, 1)
    query = torch.rand(count, 1)*2-1
    context = torch.cat([x1, a*x1+b, x2, a*x2+b, query], 1)
    return context, a*query+b
model = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
for _ in range(500):
    x, y = tasks(128)
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
model.eval()
with torch.no_grad():
    x, y = tasks(100)
    print('Ошибка на новых правилах без дообучения:', F.mse_loss(model(x), y).item())
''', 'На проверке меняются правила задач, но optimizer больше не вызывается. Модель использует две пары из контекста, чтобы ответить на новый запрос.', 'Обнули значения ответов в контексте. Сравни ошибку: это проверяет, использует ли сеть предоставленные примеры.', 'Предварительное обучение учит модель читать описание задачи из входа. Во время применения обучение весов не происходит: новые примеры меняют только контекст.', 'Это фиксированный числовой контекст маленькой сети, а не few-shot prompting LLM. Он показывает различие между изменением входа и изменением весов.')

add('preferences', 'Обучить предпочтение одному ответу над другим через парную ошибку DPO.', '''
logits = nn.Parameter(torch.zeros(2))
reference = torch.zeros(2).log_softmax(0).detach()
optimizer = torch.optim.Adam([logits], lr=0.05)
for _ in range(100):
    logp = logits.log_softmax(0)
    # В каждой паре ответ 1 предпочитается ответу 0.
    advantage = (logp[1]-logp[0])-(reference[1]-reference[0])
    loss = -F.logsigmoid(0.5*advantage)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Вероятности ответов:', logits.softmax(0).detach().tolist())
''', 'Вероятность предпочитаемого ответа 1 растёт относительно исходной политики.', 'Добавь часть пар с противоположным предпочтением. Что происходит при равных долях противоречивых оценок?', 'Модель сравнивает, насколько она предпочитает выбранный ответ отвергнутому. Из этого вычитается такое же предпочтение фиксированной исходной модели.', 'Показана парная DPO-ошибка для двух готовых действий. Для языка используются суммы логарифмов вероятностей токенов; хорошие оценки людей и проверка качества остаются отдельной задачей.')

add('bandits', 'Выбирать одно из трёх действий, постепенно выясняя, какое чаще приносит награду.', '''
true_chance = torch.tensor([0.2, 0.5, 0.8])
success = torch.ones(3); failure = torch.ones(3)
count = torch.zeros(3, dtype=torch.long)
for _ in range(800):
    guesses = torch.distributions.Beta(success, failure).sample()
    action = int(guesses.argmax())
    reward = int(torch.rand(()) < true_chance[action])
    success[action] += reward; failure[action] += 1-reward
    count[action] += 1
print('Сколько раз выбрано каждое действие:', count.tolist())
print('Оценки вероятностей:', (success/(success+failure)).tolist())
''', 'Третье действие выбирается чаще, но другие сначала исследуются. Оценка по редкому действию менее надёжна.', 'На середине опыта поменяй вероятности первого и третьего действий. Насколько быстро метод замечает смену?', 'Thompson sampling хранит распределение неопределённости для каждого действия. Случайная проба из распределения сочетает исследование и использование уже хороших вариантов.', 'Это стационарный бандит без контекста и отложенных последствий. Для меняющихся условий нужна адаптация; для длинных цепочек решений — обучение с подкреплением.')

add('rl lab-game', 'Обучить маленькую сеть выбирать шаг в коридоре из пяти клеток, чтобы добраться до награды.', '''
model = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
# Собираем все переходы маленькой среды. Действие 0 — влево, 1 — вправо.
states, actions, next_states, rewards, done = [], [], [], [], []
for state in range(4):
    for action in range(2):
        nxt = max(0, min(4, state+(1 if action else -1)))
        states.append(state); actions.append(action); next_states.append(nxt)
        rewards.append(1.0 if nxt==4 else -0.02); done.append(nxt==4)
s = F.one_hot(torch.tensor(states), 5).float()
ns = F.one_hot(torch.tensor(next_states), 5).float()
a = torch.tensor(actions); reward = torch.tensor(rewards); terminal = torch.tensor(done)
for _ in range(300):
    with torch.no_grad():
        target = reward+0.95*model(ns).max(1).values*(~terminal)
    value = model(s).gather(1, a[:, None]).squeeze(1)
    loss = F.mse_loss(value, target)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
with torch.no_grad():
    print('Действия в клетках 0–3:', model(torch.eye(5)[:4]).argmax(1).tolist())
''', 'Ожидаются действия [1, 1, 1, 1]: двигаться вправо к награде. Ошибка обучения сравнивает ценность действия с наградой и будущей ценностью.', 'Перенеси награду в клетку 0 и исключи её из обучаемых состояний. Переобучи политику и сравни маршрут.', 'Сеть оценивает пользу каждого действия в состоянии. Цель складывается из немедленной награды и оценки лучшего следующего действия; градиент не проходит через эту цель.', 'Это fitted Q-learning на полном наборе переходов игрушечной среды. Полный DQN обычно требует replay buffer, отдельной целевой сети и исследования действий. На реальных средах этот краткий код не достаточен.')

add('policy', 'Обучить вероятностный выбор действия непосредственно по награде.', '''
logits = nn.Parameter(torch.zeros(3))
optimizer = torch.optim.Adam([logits], lr=0.04)
reward_table = torch.tensor([0., 0.4, 1.])
for _ in range(250):
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample((64,))
    rewards = reward_table[actions]
    advantage = rewards-rewards.mean()
    loss = -(distribution.log_prob(actions)*advantage.detach()).mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Вероятности действий:', logits.softmax(0).detach().tolist())
''', 'Политика постепенно отдаёт почти всю вероятность третьему действию с самой высокой наградой.', 'Сделай награду случайной. Сравни обучение с вычитанием среднего rewards.mean() и без него.', 'Действия, давшие результат выше среднего, становятся вероятнее. Ниже среднего — менее вероятными. Ошибка использует логарифм вероятности уже выбранного действия.', 'Это REINFORCE на одном шаге, поясняющий градиент политики. Современные алгоритмы вроде PPO добавляют оценку ценности и ограничения обновления; их здесь нет.')

add('world', 'Обучить модель следующего состояния и с её помощью выбрать действие к цели.', '''
state = torch.randn(400, 1)
action = torch.rand(400, 1)*2-1
next_state = state+0.3*action
model = nn.Linear(2, 1)
fit_regression(model, torch.cat([state[:320], action[:320]], 1), next_state[:320])
with torch.no_grad():
    candidates = torch.linspace(-1, 1, 21)[:, None]
    current = torch.full_like(candidates, 0.8)
    forecast = model(torch.cat([current, candidates], 1))
    chosen = candidates[forecast.square().argmin()]
    error = F.mse_loss(model(torch.cat([state[320:], action[320:]], 1)), next_state[320:])
print('Ошибка модели мира:', error.item(), 'действие к нулю:', chosen.item())
''', 'Модель предсказывает движение и выбирает отрицательное действие из положения 0.8, чтобы приблизиться к нулю.', 'Планируй на пять шагов вперёд, подставляя предсказания обратно. Как ошибка модели накапливается в длинном прогнозе?', 'Модель мира учится отвечать «что случится, если сделать это?». Планировщик перебирает действия внутри предсказанной среды и выбирает подходящее.', 'Это одномерная динамика без изображений и скрытого состояния. Современные world models значительно сложнее; хорошее предсказание одного шага не гарантирует хороший план.')

add('selfplay', 'Обучать стратегию против собственного среднего прошлого поведения в игре «камень, ножницы, бумага».', '''
payoff = torch.tensor([[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]])
counts = torch.tensor([9., 3., 1.])
for _ in range(2000):
    opponent = counts/counts.sum()
    # Лучший ответ на среднюю стратегию собственных прошлых игр.
    action = int((payoff@opponent).argmax())
    counts[action] += 1
average = counts/counts.sum()
print('Средняя стратегия:', average.tolist())
print('Выгода лучшего ответа соперника:', (payoff@average).max().item())
assert (payoff@average).max().item() < 0.05
''', 'Средняя стратегия приближается к равным вероятностям 1/3. Соперник почти не может её использовать: выгода лучшего ответа меньше 0.05.', 'Замени среднюю стратегию на одно последнее действие. Посмотри, почему получается бесконечный цикл ответов.', 'Каждая новая версия стратегии отвечает на накопленное поведение прежних версий. История не позволяет забыть старые способы игры.', 'Это fictitious play в маленькой матричной игре. Здесь нет нейросети, поиска по дереву и полного алгоритма AlphaZero.', kind='Демонстрация обучения стратегии')

add('active', 'Выбрать объекты, ответы на которые полезнее всего запросить у человека.', '''
x, y = points(320)
model = nn.Linear(2, 2)
fit_classifier(model, x[:20], y[:20])
with torch.no_grad():
    probability = model(x[20:240]).softmax(1)
    entropy = -(probability*probability.clamp_min(1e-8).log()).sum(1)
    request = entropy.topk(20).indices+20
print('Номера объектов для разметки:', request.tolist())
fit_classifier(model, torch.cat([x[:20], x[request]]), torch.cat([y[:20], y[request]]))
print('После дополнительных 20 меток:', accuracy(model, x[240:], y[240:]))
''', 'Запрашиваются 20 неоднозначных объектов. Новое качество нужно сравнить с добавлением 20 случайных меток при том же бюджете.', 'Добавь случайную стратегию как контроль и повтори опыт на нескольких seed.', 'Энтропия измеряет, насколько распределён ответ между категориями. При двух почти равных вероятностях модель просит подсказку.', 'Неуверенные объекты могут быть шумом или ошибками данных. Активное обучение требует контроля качества разметки и разнообразия запросов.')

add('continual', 'Измерить забывание старой задачи и влияние повторения старых примеров.', '''
x = torch.randn(400, 2)
y_a = (x[:,0]>0).long(); y_b = (x[:,1]>0).long()
# Третий признак сообщает, какая именно задача запрошена.
a = torch.cat([x, torch.zeros(400, 1)], 1)
b = torch.cat([x, torch.ones(400, 1)], 1)
first = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, 2))
fit_classifier(first, a[:300], y_a[:300])
naive, replay = copy.deepcopy(first), copy.deepcopy(first)
fit_classifier(naive, b[:300], y_b[:300])
fit_classifier(replay, torch.cat([b[:300], a[:80]]), torch.cat([y_b[:300], y_a[:80]]))
print('Старая задача после обычного дообучения:', accuracy(naive, a[300:], y_a[300:]))
print('Старая задача с повторением:', accuracy(replay, a[300:], y_a[300:]))
''', 'Повторение старых примеров обычно уменьшает забывание. Качество новой задачи тоже нужно измерять.', 'Сравни память из 5, 20 и 80 старых примеров. Как меняется компромисс между старым и новым?', 'Последовательное обучение сдвигает веса под свежие данные. Небольшая память прошлых данных напоминает модели о старых зависимостях.', 'Показан replay с известным идентификатором задачи. Поток без таких меток, приватность данных и большие распределительные сдвиги требуют других решений.')

add('federated', 'Обучить локальные копии модели на двух наборах и усреднить их параметры.', '''
x, y = points(400)
global_model = nn.Linear(2, 2)
clients = [(x[:150], y[:150]), (x[150:300], y[150:300])]
for _ in range(15):
    local_models = []
    for local_x, local_y in clients:
        local = copy.deepcopy(global_model)
        fit_classifier(local, local_x, local_y, steps=5)
        local_models.append(local)
    with torch.no_grad():
        for name, parameter in global_model.named_parameters():
            parameter.copy_(torch.stack([dict(m.named_parameters())[name] for m in local_models]).mean(0))
print('Качество общей модели:', accuracy(global_model, x[300:], y[300:]))
''', 'Общая модель учится через обмен весами, хотя каждый локальный шаг видел только свой набор.', 'Раздай клиентам объекты разных категорий. Сравни устойчивость с равномерным разделением.', 'Каждый клиент начинает с общей модели, делает несколько шагов и возвращает веса. Сервер объединяет их; при разных размерах наборов нужно взвешенное среднее.', 'Это симуляция FedAvg в одном процессе. Данные не передаются между локальными шагами, но сами обновления могут раскрывать информацию; приватность этим кодом не обеспечивается.')

add('curriculum', 'Обучать сначала на простых объектах, затем постепенно добавлять сложные.', '''
x, y = points(400)
margin = (x[:,0]+0.5*x[:,1]).abs()
order = margin[:300].argsort(descending=True)
model = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
for size in [50, 120, 300]:
    chosen = order[:size]
    fit_classifier(model, x[chosen], y[chosen], steps=50)
    print('Учебных объектов:', size, 'качество:', accuracy(model, x[300:], y[300:]))
''', 'Программа показывает качество после каждого расширения учебной части. Пользу порядка нужно сравнивать с перемешиванием при том же числе обновлений.', 'Обрати порядок: сначала ближайшие к границе точки. Измени шум меток и сравни результат.', 'Далеко от границы правильный ответ легче определить. Постепенное усложнение может помочь модели сначала усвоить устойчивые закономерности.', 'Выбор «простоты» сделан по известной формуле искусственной задачи. На реальных данных её нельзя определить по недоступному правильному правилу.')

add('automl', 'Автоматически выбрать ширину небольшой сети по отдельной проверочной выборке.', '''
x, y = points(500)
results = []
for width in [4, 12, 32]:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, 2))
    fit_classifier(model, x[:300], y[:300], steps=80)
    score = accuracy(model, x[300:400], y[300:400])
    results.append((score, width, model))
score, width, best = max(results, key=lambda r:r[0])
print('Выбрана ширина:', width, 'по проверке:', score)
print('Окончательный тест:', accuracy(best, x[400:], y[400:]))
''', 'Ширина выбирается по валидации, а итоговая оценка считается на третьей, не использованной для выбора части.', 'Добавь скорость обучения и штраф весов в пространство поиска. Следи за общим бюджетом запусков.', 'Человек задаёт, какие варианты разрешено проверить и чем измерять качество. Программа обучает кандидатов и выбирает лучший по одинаковому правилу.', 'Это небольшой перебор, не полный AutoML. Слишком большой поиск может подстроиться даже под валидацию, поэтому окончательный тест хранится отдельно.')

add('distill', 'Обучить маленького ученика повторять вероятности более широкой модели.', '''
x, y = points(400)
teacher = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
fit_classifier(teacher, x[:300], y[:300])
teacher.eval().requires_grad_(False)
student = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.03)
temperature = 2.
for _ in range(180):
    with torch.no_grad():
        soft_target = (teacher(x[:300])/temperature).softmax(1)
    soft_loss = F.kl_div((student(x[:300])/temperature).log_softmax(1), soft_target, reduction='batchmean')*temperature**2
    loss = 0.7*soft_loss+0.3*F.cross_entropy(student(x[:300]), y[:300])
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
print('Учитель:', accuracy(teacher, x[300:], y[300:]))
print('Ученик:', accuracy(student, x[300:], y[300:]))
''', 'Ученик может приблизиться к учителю на этой простой задаче, имея меньше параметров. Он не обязан стать лучше учителя.', 'Сравни ученика с дистилляцией и того же ученика, обученного только по правильным категориям.', 'Учитель сообщает не только выбранный класс, но и относительную уверенность в остальных. Температура делает распределение мягче, чтобы ученик видел эти различия.', 'Это дистилляция классификатора. Качество ученика ограничено его возможностями, данными и ошибками учителя; перенос рассуждений большой модели сложнее.')

add('quant', 'Упаковать веса в 8-битные целые числа и измерить ошибку обратного восстановления.', '''
weights = torch.randn(128, 64)*0.1
scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)/127
packed = (weights/scale).round().clamp(-127, 127).to(torch.int8)
restored = packed.float()*scale
x = torch.randn(20, 64)
print('Ошибка выхода:', F.mse_loss(x@restored.T, x@weights.T).item())
print('Байт до:', weights.numel()*weights.element_size())
print('Байт после с масштабами:', packed.numel()*packed.element_size()+scale.numel()*scale.element_size())
''', 'Веса занимают около четверти исходного объёма плюс масштабы; восстановленный выход немного отличается от исходного.', 'Добавь очень большой вес в одну строку. Сравни общий масштаб всей матрицы и отдельные масштабы по строкам.', 'В каждой строке вещественные веса делятся на масштаб и округляются в диапазон int8. При восстановлении масштаб возвращает приблизительные значения.', 'Это ручная проверка квантования весов. Умножение после восстановления остаётся float32 и не ускоряется автоматически. Для реального исполнения используй актуальный torchao и поддерживаемые ядра.', kind='Демонстрация сжатия')

add('pruning', 'Обнулить малые веса и проверить, что разреженность и уменьшение файла — разные вещи.', '''
model = nn.Linear(20, 10)
x = torch.randn(30, 20)
with torch.no_grad():
    before = model(x)
    threshold = torch.quantile(model.weight.abs(), 0.5)
    model.weight.mul_(model.weight.abs()>=threshold)
    after = model(x)
print('Нулевых весов:', int((model.weight==0).sum()))
print('Плотная память весов в байтах:', model.weight.numel()*model.weight.element_size())
print('Ошибка после удаления:', F.mse_loss(after, before).item())
''', 'Около половины весов обнулены, но плотный массив хранит столько же чисел. Выход меняется.', 'Удали целые выходные каналы, создав новый меньший nn.Linear. Сравни память с простым обнулением.', 'Малые веса удаляются по порогу. Чтобы это ускоряло вычисления, нужен поддерживаемый разреженный формат или физически уменьшенная архитектура.', 'В примере веса случайные, модель не дообучается после удаления. Рабочий pruning оценивает важность на обученной модели, проверяет качество и возможность ускорения на целевом устройстве.')

add('factorweights', 'Заменить большую матрицу произведением двух узких и измерить потерю точности.', '''
weights = torch.randn(64, 4)@torch.randn(4, 32)+0.01*torch.randn(64, 32)
u, s, vh = torch.linalg.svd(weights, full_matrices=False)
rank = 4
left = u[:, :rank]*s[:rank]
right = vh[:rank]
x = torch.randn(20, 32)
print('Ошибка:', F.mse_loss((x@right.T)@left.T, x@weights.T).item())
print('Параметров до и после:', weights.numel(), left.numel()+right.numel())
''', 'Число параметров падает с 2048 до 384. Ошибка мала, потому что исходная матрица почти имеет ранг 4.', 'Подними добавленный шум до 1.0. Сравни качество для рангов 2, 4, 16.', 'SVD выделяет наиболее сильные направления матрицы. Оставив несколько направлений, получаем два последовательных небольших преобразования.', 'Случайная или реально обученная матрица не обязана хорошо сжиматься. Меньше умножений также не всегда означает меньшую задержку на устройстве.')

add('cache', 'Проверить, что сохранённые ключи и значения дают тот же attention при чтении по одному токену.', '''
x = torch.randn(1, 1, 10, 8)
q, k, v = x+0.1, x-0.1, x*0.5
full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
k_cache, v_cache, outputs = [], [], []
for t in range(10):
    k_cache.append(k[:, :, t:t+1]); v_cache.append(v[:, :, t:t+1])
    # У запроса длины 1 все сохранённые позиции уже в прошлом.
    output = F.scaled_dot_product_attention(q[:, :, t:t+1],
        torch.cat(k_cache, dim=2), torch.cat(v_cache, dim=2), is_causal=False)
    outputs.append(output)
cached = torch.cat(outputs, dim=2)
print('Совпадает с полным расчётом:', torch.allclose(full, cached, atol=1e-6))
assert torch.allclose(full, cached, atol=1e-6)
''', 'Пошаговый и полный ответы совпадают. Ключи и значения прошлых позиций повторно не вычислялись.', 'Измерь объём кэша при длинах 100, 1000, 10000. Почему ускорение генерации расходует память?', 'Кэш хранит результат уже обработанного контекста. Новый запрос обращается к старым ключам и значениям, не вычисляя их заново.', 'Здесь показана корректность attention-кэша одной головы. Нет полного генератора, позиционных преобразований и измерения скорости; конкатенация списков не является эффективной промышленной реализацией.')

add('binary', 'Заменить веса на два значения и увидеть, какую точность теряет крайнее сжатие.', '''
weights = torch.randn(32, 16)
scale = weights.abs().mean(1, keepdim=True)
binary = torch.where(weights>=0, 1., -1.)*scale
x = torch.randn(40, 16)
print('Ошибка выхода:', F.mse_loss(x@binary.T, x@weights.T).item())
print('Теоретические биты знаков:', weights.numel())
print('Фактические байты этого float32-тензора:', binary.numel()*binary.element_size())
''', 'Ошибка заметнее, чем при int8. Теоретическое представление использует один бит знака на вес, но этот код хранит восстановленные float32.', 'Сравни ошибку с 8-битным примером. Какие задачи могут терпеть такое сильное упрощение?', 'Каждый вес сохраняет только знак; средний модуль по строке задаёт общий масштаб. Сеть теряет различия между весами одного знака.', 'Это демонстрация бинаризации, не готовая BitNet или обученная бинарная сеть. Реальная экономия требует упаковки битов, подходящих операций и обычно специального обучения.')

add('mac', 'Выбрать доступное устройство Mac и выполнить на нём один настоящий шаг обучения.', '''
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = nn.Linear(2, 1).to(device)
x = torch.randn(64, 2, device=device)
y = x.sum(1, keepdim=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
for _ in range(150):
    loss = F.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
if device.type == 'mps':
    torch.mps.synchronize()
print('Устройство:', device, 'ошибка:', loss.detach().cpu().item())
''', 'На совместимом Mac выводится mps; иначе cpu. Модель и данные находятся на одном устройстве.', 'Сравни время для 64 и 65536 строк, синхронизируя устройство перед окончанием замера. На маленьком примере GPU может быть медленнее.', 'MPS позволяет PyTorch использовать GPU Apple. Проверка доступности обеспечивает запасной путь; перенос на устройство нужен и для модели, и для тензоров.', 'Набор поддерживаемых операций и реальная скорость зависят от версии macOS, оборудования и PyTorch. Этот пример не является сравнением PyTorch и MLX.')

add('export', 'Экспортировать вычисление через актуальный torch.export и проверить совпадение результата.', '''
model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2)).eval()
x = torch.randn(3, 4)
program = torch.export.export(model, (x,))
with torch.no_grad():
    original = model(x)
    exported = program.module()(x)
print('Совпадение:', torch.allclose(original, exported))
print('Узлов графа:', len(list(program.graph.nodes)))
assert torch.allclose(original, exported)
# Для файла можно вызвать torch.export.save(program, 'model.pt2').
''', 'Экспортированный граф даёт тот же результат для входа подходящего размера.', 'Подай 5 строк вместо 3. Разберись, как задать динамический размер через torch.export.Dim.', 'Экспорт захватывает операции модели и ограничения входов в отдельное представление. Сравнение результатов проверяет, что вычисление не изменилось.', 'Этот пример не гарантирует запуск на телефоне или в Rust. Для целевого устройства нужен поддерживаемый runtime и соответствующая цепочка преобразования. Старый TorchScript здесь не используется.')

add('repro', 'Повторить обучение с одинаковой и разной инициализацией и сравнить ответы.', '''
x = torch.randn(100, 2); y = x.sum(1, keepdim=True)
def run(seed):
    torch.manual_seed(seed)
    model = nn.Linear(2, 1)
    fit_regression(model, x, y, steps=20)
    return model(x).detach()
a, b, c = run(7), run(7), run(8)
print('Одинаковый seed:', torch.equal(a, b))
print('Разница при другом seed:', (a-c).abs().max().item())
print('Версия PyTorch:', torch.__version__)
assert torch.equal(a, b)
''', 'В одном CPU-окружении одинаковый seed повторяет результат; другой seed меняет короткий недообученный запуск.', 'Увеличь обучение до 300 шагов. Сходятся ли результаты разных начальных состояний ближе друг к другу?', 'Seed задаёт последовательность псевдослучайных чисел. Для воспроизводимости также фиксируются данные, версии библиотек, разделение выборок и параметры опыта.', 'Seed не гарантирует побитовое совпадение между разными устройствами, операционными системами и версиями библиотек.')

add('rag lab-search', 'Найти нужный фрагмент собственной базы и вернуть ответ вместе с источником.', '''
documents = ['лимит загрузки файла 20 мегабайт', 'пароль можно изменить в настройках',
             'экспорт заметок доступен в меню холста']
vocab = {w:i for i,w in enumerate(sorted(set(' '.join(documents).split())))}
def vector(text):
    result = torch.zeros(len(vocab))
    for word in text.split():
        if word in vocab: result[vocab[word]] += 1
    return F.normalize(result, dim=0)
query = 'какой лимит загрузки файла'
scores = torch.stack([vector(d) for d in documents])@vector(query)
best = int(scores.argmax())
if scores[best] < 0.2:
    print('В базе нет достаточного основания для ответа.')
else:
    print('Найденный фрагмент:', documents[best])
    print('Источник: документ', best+1)
''', 'Находится документ с лимитом 20 мегабайт и возвращается ссылка на его номер. На неизвестную тему с нулевым сходством — отказ.', 'Добавь два противоречивых документа с разными датами. Как нужно изменить выдачу, чтобы не скрыть противоречие?', 'Поиск выбирает опорный фрагмент из доступных документов. В полноценной RAG-системе языковая модель получает эти фрагменты и формулирует ответ с проверяемыми ссылками.', 'Это прозрачный лексический поиск и извлечение, не полноценная генеративная RAG. Для смыслового поиска заменяют кодирование на актуальный encoder; качество поиска и обоснованность ответа проверяют раздельно.')

add('agents', 'Собрать ограниченный цикл «выбрать инструмент → получить наблюдение → ответить».', '''
def lookup_limit():
    return {'value': 20, 'unit': 'MB', 'source': 'настройки проекта'}
tools = {'lookup_limit': lookup_limit}
# План в этом опыте задан явно, чтобы видеть границу действий.
plan = [{'tool': 'lookup_limit'}]
observations = []
for step in plan[:3]:
    name = step['tool']
    if name not in tools:
        raise ValueError('Неизвестный инструмент')
    observations.append(tools[name]())
fact = observations[-1]
print(f"Лимит: {fact['value']} {fact['unit']}. Источник: {fact['source']}.")
''', 'Ответ строится из результата инструмента. Никакие команды оболочки и произвольный код из плана не исполняются.', 'Добавь неизвестное имя инструмента и ошибку внутри lookup_limit. Сделай понятный ответ вместо бесконечного повторения.', 'Агентная система связывает выбор следующего шага с исполнением разрешённых инструментов и чтением результата. Модель может предлагать шаг, но исполняющая программа проверяет его.', 'Это каркас цикла с заданным планом; здесь нет модели-планировщика. Он показывает устройство системы, а не обучение отдельной «агентной нейросети».', kind='Демонстрация системы')

add('kg', 'Ответить на вопрос по цепочке явных связей в графе знаний.', '''
facts = [('камера', 'часть', 'робот'), ('робот', 'находится', 'цех'),
         ('датчик', 'часть', 'станок'), ('станок', 'находится', 'склад')]
object_name = 'камера'
parents = [b for a, relation, b in facts if a==object_name and relation=='часть']
places = [b for a, relation, b in facts if a in parents and relation=='находится']
print('Где находится камера:', places)
print('Основание:', [f for f in facts if f[0] in [object_name]+parents])
assert places == ['цех']
''', 'Ответ «цех» сопровождается двумя использованными фактами: камера — часть робота, робот — в цехе.', 'Добавь дату каждому факту и два разных местоположения робота. Определи правило выбора актуального.', 'Граф хранит объекты и отношения явно. Запрос проходит по разрешённой цепочке связей, поэтому основание ответа можно показать пользователю.', 'Это точный запрос к маленькому графу, без автоматического извлечения фактов и без нейросетевого вывода. Надёжность ответа ограничена качеством и актуальностью записей.')

add('arrays', 'Разобраться в форме тензора и увидеть, как пакет объектов проходит линейный слой.', '''
x = torch.tensor([[1.,2.,3.], [4.,5.,6.]])
weights = torch.tensor([[1.,0.], [0.,1.], [1.,1.]])
answer = x@weights
print('Вход:', tuple(x.shape), 'веса:', tuple(weights.shape))
print('Ответ:', tuple(answer.shape), answer.tolist())
assert answer.tolist() == [[4.,5.], [10.,11.]]
''', 'Два объекта с тремя признаками превращаются в два объекта с двумя выходами. Форма ответа — (2, 2).', 'Добавь третью строку входа. Затем поменяй число столбцов и объясни сообщение о несовместимых размерах.', 'Тензор — массив чисел с формой. В умножении (число объектов, признаки) × (признаки, выходы) внутренние размеры должны совпадать.', 'Это базовая операция, используемая и в современных больших моделях. Само умножение матриц ещё не обучает веса.')

add('derivatives', 'Проверить производную через автоматическое дифференцирование и маленькое изменение входа.', '''
x = torch.tensor(3., requires_grad=True)
y = x.square()+2*x
y.backward()
h = 0.001
finite_difference = ((x.detach()+h).square()+2*(x.detach()+h)-y.detach())/h
print('Производная PyTorch:', x.grad.item())
print('Численная оценка:', finite_difference.item())
assert abs(x.grad.item()-8)<1e-6
''', 'Точная производная в точке 3 равна 8. Численная оценка близка к ней, но содержит погрешность шага и округления.', 'Сравни h=0.1 и h=0.000001. Почему слишком маленький шаг тоже может ухудшить оценку?', 'Производная говорит, насколько изменится выход при малом изменении входа. PyTorch хранит цепочку операций и проходит её назад, применяя правило цепочки.', 'Для всей сети действуют те же правила, но градиент не гарантирует глобальный минимум. Также важно, по какой переменной вычисляется производная.')

add('optimization', 'Проследить, как размер шага влияет на поиск минимума ошибки.', '''
for learning_rate in [0.05, 0.4, 1.1]:
    parameter = nn.Parameter(torch.tensor(5.))
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    losses = []
    for _ in range(20):
        loss = (parameter-2).square()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    print('Шаг:', learning_rate, 'первые ошибки:', losses[:3], 'последняя:', losses[-1])
''', 'Умеренный шаг приближает параметр к 2. Слишком большой шаг заставляет перескакивать минимум и увеличивать ошибку.', 'Добавь моментум или замени SGD на AdamW. Сравни поведение на этой же функции, сохранив начальную точку.', 'Градиент указывает направление роста ошибки. Оптимизатор двигает параметр в противоположную сторону, а learning rate задаёт размер движения.', 'SGD здесь выбран для прозрачности формулы. Для современных моделей оптимизатор, расписание шага и регуляризацию выбирают под задачу; один опыт не устанавливает лучший оптимизатор.')

add('data split', 'Отделить подготовку данных от проверки, чтобы будущие значения не влияли на обучение.', '''
x = torch.randn(300, 3)
x[240:] += 1.5  # Более поздние данные пришли из других условий.
train, validation, test = x[:180], x[180:240], x[240:]
mean = train.mean(0); std = train.std(0).clamp_min(1e-6)
train_scaled = (train-mean)/std
test_scaled = (test-mean)/std
print('Среднее учебной части:', train_scaled.mean(0).tolist())
print('Среднее теста с учебными настройками:', test_scaled.mean(0).tolist())
assert train_scaled.mean(0).abs().max()<1e-6
''', 'Учебная часть имеет среднее около нуля. Тест остаётся сдвинутым: это честно отражает изменение условий.', 'Вычисли mean по всем 300 строкам и сравни. Почему такая подготовка уже использовала сведения о тесте?', 'Настройки нормализации учатся только на учебной части. Валидация помогает выбирать модель, а тест служит финальной проверкой; их нельзя незаметно подмешивать.', 'Разбиение по строкам подходит не всем данным. Для пациентов, пользователей, устройств и временных рядов нужно разделять по соответствующим группам или времени.')

add('metrics', 'Увидеть, как 95% точности могут скрывать полный провал поиска редких событий.', '''
y = torch.tensor([0]*95+[1]*5)
prediction = torch.zeros_like(y)
true_positive = ((prediction==1)&(y==1)).sum()
false_positive = ((prediction==1)&(y==0)).sum()
false_negative = ((prediction==0)&(y==1)).sum()
accuracy_value = prediction.eq(y).float().mean()
precision = true_positive/(true_positive+false_positive).clamp_min(1)
recall = true_positive/(true_positive+false_negative).clamp_min(1)
print('Общая точность:', accuracy_value.item())
print('Точность тревог:', precision.item(), 'доля найденных событий:', recall.item())
''', 'Общая точность равна 0.95, а доля найденных редких событий — 0. Нулевой precision здесь является принятой условностью при отсутствии тревог.', 'Создай модель, которая выдаёт 10 тревог и находит 4 из 5 событий. Посчитай цену пропусков и ложных тревог.', 'Accuracy считает все верные ответы. Recall отвечает, сколько нужных событий нашли. Precision — сколько тревог оказались настоящими. Эти вопросы не взаимозаменяемы.', 'Метрику выбирают по последствиям ошибок. Одной цифры без состава данных и порога принятия решения недостаточно.')

add('generalization', 'Сравнить обучение по настоящим и случайно перемешанным ответам.', '''
x, y = points(400)
for random_labels in [False, True]:
    labels = y[:300].clone()
    if random_labels: labels = labels[torch.randperm(len(labels))]
    model = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 2))
    fit_classifier(model, x[:300], labels, steps=200)
    print('Случайные метки:', random_labels,
          'учебная часть:', accuracy(model, x[:300], labels),
          'новые объекты:', accuracy(model, x[300:], y[300:]))
''', 'Случайные метки не содержат полезного правила для новых объектов. Улучшение учебной оценки не означает улучшение настоящей задачи.', 'Увеличь сеть и число шагов на случайных метках. Следи отдельно за учебной и проверочной оценками.', 'Обобщение означает перенос закономерности на новые объекты. Случайные ответы можно частично запомнить, но из них нельзя восстановить исходное правило.', 'Этот опыт не покрывает все виды переобучения. Настоящие данные могут содержать утечки, повторяющиеся объекты и неверное разделение по времени.')

add('uncertainty', 'Сравнить ответы нескольких моделей на знакомых и далёких от обучения объектах.', '''
x = torch.linspace(-1, 1, 80)[:, None]; y = x.sin()
query = torch.tensor([[-3.], [0.], [3.]])
predictions = []
for seed in range(5):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    fit_regression(model, x, y)
    with torch.no_grad(): predictions.append(model(query))
answers = torch.stack(predictions)
print('Средние ответы:', answers.mean(0).squeeze().tolist())
print('Разброс между моделями:', answers.std(0).squeeze().tolist())
''', 'В центре учебного диапазона модели обычно согласны сильнее, чем за его пределами. Низкий разброс всё равно не является гарантией правильности.', 'Обучи все пять моделей с одинаковым seed. Почему исчезновение разнообразия создаёт ложное ощущение уверенности?', 'Ансамбль обучается несколько раз с разной инициализацией. Расхождение ответов показывает чувствительность результата к выбранной модели.', 'Это эвристика неопределённости модели, а не калиброванный доверительный интервал и не полная оценка шума данных.')

add('explain', 'Измерить, насколько ухудшается качество, если перемешать один признак между объектами.', '''
x = torch.randn(400, 3); y = (2*x[:,0]+x[:,1]>0).long()
model = nn.Linear(3, 2)
fit_classifier(model, x[:300], y[:300])
baseline = accuracy(model, x[300:], y[300:])
for feature in range(3):
    broken = x[300:].clone()
    broken[:, feature] = broken[torch.randperm(100), feature]
    print('Признак', feature, 'падение качества:', baseline-accuracy(model, broken, y[300:]))
''', 'Перемешивание первого признака обычно вредит сильнее всего; третий почти не влияет, потому что не участвовал в правильном ответе.', 'Сделай третий признак копией первого. Почему важность каждого по отдельности может измениться?', 'Перемешивание сохраняет значения признака, но разрушает связь с конкретным объектом. Падение качества показывает, насколько модель полагалась на эту связь.', 'Permutation importance объясняет поведение модели на выбранных данных. Она не доказывает причинное влияние признака, особенно при сильных корреляциях.')
