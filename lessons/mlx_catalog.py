"""Native MLX counterparts; no PyTorch import or bridge."""
from textwrap import dedent
from catalog import LESSONS
from display_source import listing, MLX_RUNTIMES

MLX = {}
EXPLANATIONS = {
    'knn': 'Массив distances хранит квадраты расстояний от новых объектов до учебных. '
           'Функция mx.argsort возвращает индексы в порядке возрастания расстояния; '
           'срез [:, :3] выбирает трёх ближайших соседей. Их категории голосуют за ответ.',
    'derivatives': 'Производная показывает, насколько изменится выход при малом изменении входа. '
                   'Вызов mx.grad(f) создаёт функцию производной для f; затем вычисляем её в точке x.',
    'mac': 'MLX размещает новые массивы на устройстве, выбранном через mx.set_default_device. '
           'Вызов mx.synchronize() дожидается окончания расчётов. Модель и данные должны '
           'находиться на одном устройстве; недоступное устройство вызывает понятную ошибку.',
    'lora': 'Основная матрица base хранит неизменное преобразование восьми входных чисел '
            'в восемь выходных. Знак @ означает матричное умножение. Модель Adapter содержит '
            'только две обучаемые матрицы, self.a и self.b: первая сжимает восемь чисел '
            'до двух, вторая возвращает восемь. Их результат прибавляется к основному '
            'ответу. Функция fit обновляет параметры модели, а base находится вне неё.',
}
EXERCISES = {
    'transformer': 'Функция attention вычисляет внимание: взвешивает сведения других позиций '
                   'по их сходству с запросом. В двух вызовах, создающих answer и changed, '
                   'замени последний аргумент True на False. Он задаёт запрет на будущие '
                   'позиции. Теперь запрет снят: проверь, меняются ли ранние ответы '
                   'после изменения последнего элемента.',
    'derivatives': 'В переменной numerical вычисляется численная оценка производной. '
                   'Число 0.001 — расстояние между двумя точками. Замени его в числителе '
                   'и знаменателе сначала на 0.1, затем на 0.000001. Почему слишком '
                   'малый шаг усиливает влияние ошибок округления?',
    'optimization': 'Здесь веса обновляются вручную. Добавь накопление предыдущего '
                    'направления: до цикла задай velocity = 0, на каждом шаге вычисляй '
                    'velocity = 0.9 * velocity + gradient и вычитай rate * velocity '
                    'из parameter. Переменная velocity хранит накопленное направление, '
                    'gradient — текущий градиент, rate — размер шага. Сравни уменьшение '
                    'ошибки с исходным вариантом при одинаковой начальной точке.',
    'mac': 'Сравни время обучения на 64 и 65536 строках. Перед началом и концом замера '
           'вызови mx.synchronize(): эта функция ждёт окончания вычислений. '
           'Проверь, когда ускорение расчётов покрывает затраты на их запуск.',
}
PREFIX = '''# Python 3.14 · MLX 0.32.4
import argparse
import platform
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

parser = argparse.ArgumentParser(description="AI Atlas / MLX")
parser.add_argument("--device", choices=["cpu", "metal", "auto"], default="cpu")
requested = parser.parse_args().device
metal_available = platform.system() == "Darwin" and mx.device_count(mx.gpu) > 0
if requested == "auto":
    requested = "metal" if metal_available else "cpu"
if requested == "metal" and not metal_available:
    parser.error("Metal недоступен. Запусти нативно на Apple silicon вне Linux-контейнера либо явно выбери --device cpu.")
mx.set_default_device(mx.gpu if requested == "metal" else mx.cpu)
mx.random.seed(7)
print("MLX | устройство:", requested, "| dtype: float32")
'''
TRAIN = '''
def fit(model, x, y, steps=160, classification=False, rate=0.02):
    def loss_fn(model, x, y):
        prediction = model(x)
        return (nn.losses.cross_entropy(prediction, y, reduction="mean") if classification
                else mx.mean(mx.square(prediction-y)))
    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=rate, bias_correction=True)
    for _ in range(steps):
        loss, grads = value_and_grad(model, x, y)
        optimizer.update(model, grads)
        # MLX откладывает расчёт до запроса результата.
        # Здесь выполняем накопленные вычисления весов, состояния и ошибки.
        mx.eval(model.parameters(), optimizer.state, loss)
    return loss.item()
'''

def add(ids, code, *, train=False, limits=None):
    for topic in ids.split():
        source = LESSONS[topic]
        MLX[topic] = dict(goal=source['goal'], code=PREFIX+(TRAIN if train else '')+'\n'+dedent(code).strip()+'\n',
            explanation=('MLX создаёт массивы на выбранном устройстве и откладывает расчёты до запроса результата. '
                         + ('Вызов mx.eval в цикле обучения выполняет накопленные вычисления. ' if train else '')
                         + EXPLANATIONS.get(topic, source['explanation'])),
            expected=source['expected'], exercise=EXERCISES.get(topic, source['exercise']), kind=source['kind'],
            limits=limits or source['limits'],
            parity='Та же учебная задача. MLX использует собственный генератор случайных чисел; численные результаты и скорость могут отличаться.')
        MLX[topic]['display'] = listing(
            '# Python 3.14 · MLX 0.32.4\nimport mlx.core as mx\nimport mlx.nn as nn\n'
            'import mlx.optimizers as optim\nfrom mlx.utils import tree_flatten\n',
            (TRAIN if train else '')+'\n'+dedent(code).strip(), MLX_RUNTIMES)
        MLX[topic]['code'] = PREFIX+'\n'+MLX[topic]['display']['body']
        MLX[topic]['devices'] = ['cpu'] if 'stream=mx.cpu' in code else ['cpu', 'metal']
        MLX[topic]['deviceNote'] = 'В этом опыте используются операции линейной алгебры, для которых выбран явный запуск на CPU.'
        if topic == 'mac':
            MLX[topic]['expected'] = 'В начале выводится выбранное устройство: CPU или Metal. Ошибка обученной модели должна быть меньше 0.05.'

add('linear classification lab-table', '''
x = mx.random.normal((300, 2))
y = 3*x[:, :1]-2*x[:, 1:]+0.7+0.1*mx.random.normal((300, 1))
model = nn.Linear(2, 1)
fit(model, x[:240], y[:240], steps=300)
error = mx.mean(mx.square(model(x[240:])-y[240:])).item()
print("Ошибка новых строк:", error, "коэффициенты:", model.weight.tolist(), "сдвиг:", model.bias.item())
assert error < 0.06
''',train=True)

add('mlp', '''
x = mx.random.uniform(-1, 1, (500, 2))
y = (mx.sum(mx.square(x), axis=1) < 0.45).astype(mx.int32)
model = nn.Sequential(nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 2))
fit(model, x[:400], y[:400], classification=True)
accuracy = mx.mean(mx.argmax(model(x[400:]), axis=1) == y[400:]).item()
print("Доля верных ответов на новых точках:", accuracy)
assert accuracy > 0.9
''',train=True)

add('supervised scratch', '''
x = mx.random.normal((320, 2))
y = (x[:, 0]+0.5*x[:, 1] > 0).astype(mx.int32)
model = nn.Sequential(nn.Linear(2, 12), nn.SiLU(), nn.Linear(12, 2))
print("До:", mx.mean(mx.argmax(model(x[240:]), axis=1) == y[240:]).item())
loss = fit(model, x[:240], y[:240], classification=True)
accuracy = mx.mean(mx.argmax(model(x[240:]), axis=1) == y[240:]).item()
print("После:", accuracy, "последняя учебная ошибка:", loss)
assert accuracy > 0.9
''',train=True)

add('cnn vision lab-vision', '''
# MLX использует NHWC: пакет, высота, ширина, каналы.
y = (mx.arange(240) % 2).astype(mx.int32)
positions = 1+(mx.arange(240)//2)%5
axis = mx.arange(8)
vertical = ((axis[None, :]-positions[:, None] >= 0) &
            (axis[None, :]-positions[:, None] < 2))[:, None, :]
horizontal = mx.transpose(vertical, (0, 2, 1))
x = mx.where(y[:, None, None] == 0, vertical, horizontal).astype(mx.float32)
x = x[:, :, :, None]+0.12*mx.random.normal((240, 8, 8, 1))
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.output = nn.Linear(8, 2)
    def __call__(self, x):
        return self.output(mx.mean(nn.silu(self.conv(x)), axis=(1, 2)))
model = CNN()
fit(model, x[:180], y[:180], classification=True)
accuracy = mx.mean(mx.argmax(model(x[180:]), axis=1) == y[180:]).item()
print("Верно на новых картинках:", accuracy)
print("Обучаемых чисел:", sum(p.size for _, p in tree_flatten(model.parameters())))
assert accuracy > 0.9
''',train=True)

add('lora', '''
x = mx.random.normal((300, 8))
base = mx.random.normal((8, 8))*0.2
base_before = mx.array(base)
true_update = mx.random.normal((8, 2))@mx.random.normal((2, 8))*0.1
y = x@(base+true_update)
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = mx.random.normal((8, 2))*0.01
        self.b = mx.zeros((2, 8))
    def __call__(self, x):
        # База находится вне Module и не передаётся оптимизатору.
        return x@base+x@self.a@self.b
model = Adapter()
before = mx.mean(mx.square(model(x[240:])-y[240:])).item()
fit(model, x[:240], y[:240], steps=250, rate=0.04)
after = mx.mean(mx.square(model(x[240:])-y[240:])).item()
merged = base+model.a@model.b
print("MSE до:", before, "после:", after, "базовых весов:", base.size, "обучаемых:", model.a.size+model.b.size)
assert after < before*0.02 and after < 0.001
assert mx.array_equal(base, base_before).item()
assert mx.allclose(model(x[240:]), x[240:]@merged, atol=1e-5).item()
''',train=True)

add('ae', '''
normal = mx.random.normal((300, 1))@mx.array([[1., 2., -1., 0.5]])
normal = normal+0.03*mx.random.normal(normal.shape)
model = nn.Sequential(nn.Linear(4, 1), nn.Linear(1, 4))
fit(model, normal[:240], normal[:240], steps=300)
ordinary = mx.mean(mx.square(model(normal[240:])-normal[240:])).item()
strange = mx.array([[5., -5., 5., -5.]])
unusual = mx.mean(mx.square(model(strange)-strange)).item()
print("Обычные:", ordinary, "необычный:", unusual)
assert ordinary < 0.02 and unusual > ordinary*100
''',train=True)

add('arrays', '''
x = mx.array([[1., 2., 3.], [4., 5., 6.]])
w = mx.array([[1., 0.], [0., 1.], [1., 1.]])
answer = x@w
print("Ответ:", answer.shape, answer.tolist())
assert answer.tolist() == [[4., 5.], [10., 11.]]
''')
add('derivatives', '''
def f(x): return mx.square(x)+2*x
x = mx.array(3.)
derivative = mx.grad(f)(x).item()
numerical = ((f(x+0.001)-f(x))/0.001).item()
print("Производная MLX:", derivative, "численная оценка:", numerical)
assert abs(derivative-8) < 1e-6 and abs(numerical-8) < 0.01
''')
add('optimization', '''
def loss_fn(x): return mx.square(x-2)
value_and_grad = mx.value_and_grad(loss_fn)
for rate in [0.05, 0.4, 1.1]:
    parameter, losses = mx.array(5.), []
    for _ in range(20):
        loss, gradient = value_and_grad(parameter)
        parameter = parameter-rate*gradient
        mx.eval(parameter, loss)
        losses.append(loss.item())
    print("Шаг:", rate, "первые ошибки:", losses[:3], "последняя:", losses[-1])
    assert losses[-1] < losses[0] if rate < 1 else losses[-1] > losses[0]
''')

ATTENTION = '''
def attention(q, k, v, causal=False):
    scores = (q@mx.swapaxes(k, -1, -2))/(q.shape[-1]**0.5)
    if causal:
        mask = mx.arange(k.shape[-2])[None, :] > mx.arange(q.shape[-2])[:, None]
        scores = mx.where(mask, -1e9, scores)
    return mx.softmax(scores, axis=-1)@v
'''
add('transformer', ATTENTION+'''
x = mx.random.normal((2, 6, 16))
projection = nn.Linear(16, 48, bias=False)
q, k, v = [mx.swapaxes(z.reshape(2, 6, 4, 4), 1, 2)
           for z in mx.split(projection(x), 3, axis=-1)]
answer = attention(q, k, v, True)
changed_v = mx.concatenate([v[:, :, :-1], v[:, :, -1:]+100], axis=2)
changed = attention(q, k, changed_v, True)
print("Размер ответа:", answer.shape, "ранние ответы не меняются:", mx.allclose(answer[:, :, :-1], changed[:, :, :-1]).item())
assert mx.allclose(answer[:, :, :-1], changed[:, :, :-1]).item()
''')
add('cache', ATTENTION+'''
x = mx.random.normal((1, 1, 10, 8))
q, k, v = x+0.1, x-0.1, x*0.5
full = attention(q, k, v, True)
keys, values, outputs = [], [], []
for t in range(10):
    keys.append(k[:, :, t:t+1]); values.append(v[:, :, t:t+1])
    outputs.append(attention(q[:, :, t:t+1], mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)))
cached = mx.concatenate(outputs, axis=2)
print("Совпадает:", mx.allclose(full, cached, atol=1e-5).item())
assert mx.allclose(full, cached, atol=1e-5).item()
''')
add('gnn graphsdata', '''
adjacency = mx.array([[1.,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]])
features = mx.array([[1.,0],[0,1],[1,1],[8,8]])
messages = (adjacency/mx.sum(adjacency, axis=1, keepdims=True))@features
answer = nn.Linear(2, 3, bias=False)(messages)
print("Сведения после обмена:", messages.tolist(), "форма ответа:", answer.shape)
assert mx.allclose(messages[-1], features[-1]).item()
''')
add('ssm', '''
state = mx.zeros((4,))
decay = mx.array([0.1, 0.5, 0.8, 0.95])
outputs = []
for t in range(30):
    state = decay*state+(1 if t == 3 else 0)
    outputs.append(mx.mean(state).item())
print("Ответ после импульса:", outputs[3:12])
assert outputs[3] > outputs[8] > outputs[20] > 0
''')
add('knn', '''
x = mx.random.normal((240, 2))
y = (x[:, 0]+0.5*x[:, 1] > 0).astype(mx.int32)
distances = mx.sum(mx.square(x[180:, None, :]-x[None, :180, :]), axis=2)
neighbors = mx.argsort(distances, axis=1)[:, :3]
prediction = mx.mean(y[neighbors], axis=1) >= 0.5
accuracy = mx.mean(prediction == y[180:]).item()
print("Верно на новых точках:", accuracy, "соседи:", neighbors[0].tolist())
assert accuracy > 0.8
''')
add('probability', '''
observations = mx.array([1,1,0,1,1,0,1,1])
a = 1+int(mx.sum(observations).item())
b = 1+len(observations)-(a-1)
ga = -mx.sum(mx.log(mx.random.uniform(1e-7, 1, (10000, a))), axis=1)
gb = -mx.sum(mx.log(mx.random.uniform(1e-7, 1, (10000, b))), axis=1)
samples = mx.sort(ga/(ga+gb))
low, high = samples[249].item(), samples[9749].item()
print("Оценка вероятности:", a/(a+b), "95% интервал:", [low, high])
assert 0.3 < low < 0.7 < high < 0.99
''',limits='Бета-распределение описывает неопределённость вероятности от 0 до 1. Здесь оно получается через суммы логарифмов; этот способ рассчитан на целые параметры. Квантили являются оценкой по 10 000 случайных выборок.')
add('factorweights', '''
weights = mx.random.normal((64,4))@mx.random.normal((4,32))+0.01*mx.random.normal((64,32))
# SVD выполняется явно на CPU; последующие умножения — на выбранном устройстве.
u, s, vt = mx.linalg.svd(weights, stream=mx.cpu)
left, right = u[:, :4]*s[:4], vt[:4]
x = mx.random.normal((20,32))
error = mx.mean(mx.square((x@right.T)@left.T-x@weights.T)).item()
print("Ошибка:", error, "параметров:", weights.size, "->", left.size+right.size)
assert error < 0.02
''',limits='Сингулярное разложение (SVD) явно выполняется на процессоре. Этот опыт проверяет сжатие весов; он не измеряет ускорение декомпозиции на Metal.')
add('reservoir', '''
signal = mx.sin(mx.arange(260)*0.13)
w, input_weights = mx.random.normal((24,24))*0.08, mx.random.normal((24,))*0.3
state, states = mx.zeros((24,)), []
for value in signal[:-1]:
    state = mx.tanh(w@state+input_weights*value)
    states.append(state)
x, y = mx.stack(states), signal[1:, None]
a, b = x[20:200], y[20:200]
readout = mx.linalg.solve(a.T@a+0.001*mx.eye(24), a.T@b, stream=mx.cpu)
error = mx.mean(mx.square(x[200:]@readout-y[200:])).item()
print("Ошибка на более позднем участке:", error)
assert error < 0.02
''',limits='Внутренние веса не обучаются. Решение линейной системы для выхода требует процессора; в этом варианте весь опыт запускается на CPU.')
add('compactarch specialblocks tinyml', '''
x = mx.random.normal((1,24,24,16))
regular = nn.Conv2d(16,32,3,padding=1,bias=False)
compact = nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16,bias=False), nn.Conv2d(16,32,1,bias=False))
counts = []
for name, model in [("Обычная", regular), ("Компактная", compact)]:
    count = sum(p.size for _, p in tree_flatten(model.parameters()))
    answer = model(x)
    mx.eval(answer)
    counts.append(count)
    print(name, "весов:", count, "байт:", count*4, "ответ NHWC:", answer.shape)
assert counts == [4608, 656]
''')
add('mac', '''
x = mx.random.normal((64,2))
y = x[:, :1]*2-x[:, 1:]
model = nn.Linear(2,1)
fit(model, x, y, steps=200)
mx.synchronize()
error = mx.mean(mx.square(model(x)-y)).item()
print("Ошибка:", error)
assert error < 0.05
''',train=True)
