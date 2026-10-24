"""Small, device-specific source listings for the lesson reader."""
import ast
import io
import re
import textwrap
import tokenize


# Explanations are attached to the statement that uses an operation, never to
# the entire lesson. Keep one explanation per operation in each source listing.
OPERATIONS = {
    'torch.corrcoef': 'Вычисляем корреляции Пирсона между строками матрицы.\nЧисла от -1 до 1 показывают силу и направление линейной связи.\nЭлемент [0, 1] — связь первого ряда со вторым, а не доказательство причины.',
    'torch.cdist': 'Вычисляем расстояния между каждой парой объектов из двух наборов.',
    'torch.linalg.svd': 'Сингулярное разложение (SVD) выделяет направления матрицы и их значимость.\nКрупнейшие сингулярные числа позволяют построить компактное приближение.',
    'torch.linalg.solve': 'Решаем систему A @ x = b без явного вычисления обратной матрицы.',
    'torch.linalg.lstsq': 'Подбираем коэффициенты, минимизирующие сумму квадратов невязок A @ x - b.',
    'torch.autograd.grad': 'Получаем производную выхода по входу.\ncreate_graph=True позволяет затем вычислить производную ещё раз.',
    'torch.fft.rfft': 'Переходим от действительного сигнала к его частотным компонентам.',
    'torch.fft.irfft': 'Восстанавливаем действительный сигнал из частотных компонент.',
    'torch.distributions.Beta': 'Бета-распределение описывает неопределённость вероятности от 0 до 1.',
    'torch.distributions.Categorical': 'Задаём случайный выбор одного действия по вероятностям категорий.',
    'torch.export.export': 'Записываем вычислительный граф модели для последующего выполнения.',
    'torch.quantile': 'Находим пороги, ниже которых лежат заданные доли значений.',
    'torch.outer': 'Внешнее произведение: получаем матрицу всех попарных произведений двух векторов.',
    'torch.bincount': 'Считаем, сколько раз встретился каждый целочисленный индекс.',
    'torch.stack': 'Собираем тензоры одинаковой формы в один, добавляя новое измерение.',
    'torch.cat': 'Соединяем тензоры вдоль уже существующего измерения dim.',
    'torch.randperm': 'Создаём случайную перестановку индексов для перемешивания объектов.',
    'torch.where': 'Для каждого элемента выбираем одно из двух значений по логическому условию.',
    'torch.full_like': 'Создаём тензор той же формы и заполняем его одной заданной величиной.',
    'torch.eye': 'Создаём единичную матрицу: единицы на диагонали, остальные элементы — нули.',
    'torch.randn': 'Создаём искусственные данные из нормального распределения со средним 0.',
    'torch.randn_like': 'Создаём случайный шум из нормального распределения той же формы, что исходный тензор.',
    'torch.rand': 'Генерируем случайные числа, равномерно распределённые от 0 до 1.',
    'torch.randint': 'Генерируем случайные целые числа; верхняя граница диапазона не включается.',
    'torch.tensor': 'Создаём тензор — числовой массив — из указанных значений.',
    'torch.zeros': 'Создаём тензор заданной формы и заполняем его нулями.',
    'torch.ones': 'Создаём тензор заданной формы и заполняем его единицами.',
    'torch.linspace': 'Создаём равномерную сетку между двумя границами, включая обе границы.',
    'torch.arange': 'Создаём последовательность чисел с постоянным шагом; конечная граница не включается.',
    'torch.allclose': 'Проверяем совпадение всех значений с учётом небольшой ошибки округления.',
    'torch.equal': 'Проверяем точное равенство формы и значений двух тензоров.',
    'torch.manual_seed': 'Фиксируем начальное состояние генератора, чтобы повторить случайные данные.',
    'torch.no_grad': 'Не записываем эти операции в граф для автоматического вычисления градиентов.',
    'torch.optim.AdamW': 'AdamW обновляет параметры по градиентам.\nПараметр weight_decay задаёт дополнительное уменьшение весов; 0 отключает его.',
    'torch.optim.Adam': 'Adam подбирает величину шага для каждого параметра по истории градиентов.',
    'torch.optim.SGD': 'SGD — градиентный спуск; параметр lr задаёт размер шага уменьшения ошибки.',
    'F.scaled_dot_product_attention': 'Внимание взвешивает значения V по сходству запросов Q и ключей K.',
    'F.cross_entropy': 'Сравниваем оценки классов с правильными метками; меньше — лучше.',
    'F.mse_loss': 'MSE — среднее значение квадратов разностей прогноза и правильного ответа.',
    'F.binary_cross_entropy_with_logits': 'Считаем ошибку бинарного классификатора прямо по исходным оценкам.',
    'F.kl_div': 'KL-дивергенция измеряет расхождение распределений ученика и учителя.',
    'F.normalize': 'Нормируем векторы; по умолчанию их евклидова длина становится равной 1.',
    'F.one_hot': 'Превращаем номер категории в вектор: единица на её месте, остальные — нули.',
    'F.logsigmoid': 'Устойчиво вычисляем логарифм сигмоиды, не вычисляя её отдельно.',
    'F.softplus': 'Softplus плавно преобразует число в положительное значение.',
    'F.elu': 'ELU сохраняет положительные значения и плавно преобразует отрицательные.',
    'nn.Linear': 'Линейный слой учится преобразованию y = x @ W.T + b.',
    'nn.Conv1d': 'Свёртка ищет локальные закономерности вдоль последовательности.',
    'nn.Conv2d': 'Свёртка применяет обучаемые фильтры к небольшим участкам изображения.',
    'nn.LSTM': 'LSTM передаёт скрытое состояние и память от одного шага последовательности к следующему.',
    'nn.Embedding': 'Каждому целочисленному индексу сопоставляем обучаемый вектор.',
    'nn.Parameter': 'Объявляем обучаемый параметр: оптимизатор сможет изменять его значение.',
    'nn.Buffer': 'Сохраняем тензор внутри модели без включения в обучаемые параметры.',
    'nn.ModuleList': 'Регистрируем список подмоделей, чтобы учитывались параметры каждой из них.',
    'nn.Sequential': 'Выход каждого слоя передаём следующему; порядок слоёв задаёт вычисление сети.',
    'nn.Flatten': 'Разворачиваем выбранные измерения в одно, сохраняя порядок элементов.',
    'nn.AdaptiveAvgPool1d': 'Усредняем участки последовательности до заданной длины выхода.',
    'nn.AdaptiveAvgPool2d': 'Усредняем участки карты признаков до заданной высоты и ширины.',
    'nn.SiLU': 'SiLU плавно ослабляет отрицательные значения и сохраняет большие положительные.\nТакая нелинейная функция позволяет сети учить сложные зависимости.',
    'nn.Tanh': 'Гиперболический тангенс ограничивает выход диапазоном от -1 до 1.',
    'copy.deepcopy': 'Создаём независимую копию модели, чтобы дальнейшее обучение не меняло оригинал.',
    'mx.linalg.svd': 'Сингулярное разложение (SVD) выделяет направления матрицы и их значимость.',
    'mx.linalg.solve': 'Решаем систему A @ x = b без явного вычисления обратной матрицы.',
    'mx.grad': 'Создаём функцию, вычисляющую производную исходной функции по её аргументу.',
    'mx.value_and_grad': 'Получаем функцию, возвращающую и значение ошибки, и её градиент.',
    'nn.value_and_grad': 'Вычисляем ошибку и градиенты по обучаемым параметрам модели MLX.',
    'nn.losses.cross_entropy': 'Измеряем ошибку классификации по исходным оценкам классов и верным меткам.',
    'optim.Adam': 'Adam обновляет параметры, учитывая историю их градиентов.',
    'optimizer.update': 'Применяем вычисленные градиенты к параметрам модели.',
    'mx.eval': 'MLX откладывает расчёт до запроса результата.\nЗдесь действительно выполняем накопленные вычисления.',
    'mx.synchronize': 'Дожидаемся завершения вычислений перед чтением результата или замером времени.',
    'mx.softmax': 'Преобразуем оценки в неотрицательные веса, сумма которых равна 1.',
    'mx.argsort': 'Получаем индексы элементов в порядке возрастания их значений.',
    'mx.sort': 'Упорядочиваем значения по возрастанию.',
    'mx.argmax': 'Находим индекс наибольшей оценки — выбранную моделью категорию.',
    'mx.concatenate': 'Соединяем массивы вдоль уже существующей оси.',
    'mx.stack': 'Объединяем массивы одинаковой формы, добавляя новую ось.',
    'mx.swapaxes': 'Меняем две оси местами, чтобы согласовать формы перед матричным умножением.',
    'mx.transpose': 'Переставляем оси массива в указанный порядок.',
    'mx.split': 'Разделяем массив на части вдоль выбранной оси.',
    'mx.where': 'По логической маске выбираем значение из одного из двух массивов.',
    'mx.random.normal': 'Создаём искусственные данные из нормального распределения со средним 0.',
    'mx.random.uniform': 'Генерируем значения равномерно внутри указанного интервала.',
    'mx.mean': 'Усредняем значения вдоль указанной оси или по всему массиву.',
    'mx.square': 'Возводим каждый элемент в квадрат; так получают квадраты ошибок.',
    'mx.sum': 'Суммируем значения вдоль выбранной оси или по всему массиву.',
    'mx.tanh': 'Ограничиваем каждое значение диапазоном от -1 до 1.',
    'tree_flatten': 'Разворачиваем дерево параметров модели в список пар «путь, массив».',
}
METHODS = {
    'backward': 'Вычисляем градиенты ошибки по обучаемым параметрам.',
    'zero_grad': 'Сбрасываем градиенты предыдущего шага: PyTorch иначе накапливает их.',
    'step': 'Обновляем обучаемые параметры по рассчитанным градиентам.',
    'detach': 'Отделяем тензор от графа градиентов: через эту ветку обучение не пойдёт.',
    'requires_grad_': 'Явно включаем или отключаем вычисление градиентов для этих параметров.',
    'topk': 'Находим k крайних значений и их индексы; largest=False выбирает наименьшие.',
    'softmax': 'Преобразуем оценки в веса с суммой 1 вдоль выбранного измерения.',
    'log_softmax': 'Устойчиво вычисляем логарифмы вероятностей классов из исходных оценок.',
    'unfold': 'Нарезаем последовательность на скользящие окна заданной длины.',
    'masked_fill': 'Заменяем элементы, отмеченные логической маской, указанным значением.',
    'clamp_min': 'Ограничиваем значения снизу, чтобы избежать слишком малого знаменателя или log(0).',
    'reshape': 'Меняем форму массива, сохраняя число и порядок элементов.',
}


def separate_statements(code):
    """Give simple statements their own lines without changing inline suites."""
    tree = ast.parse(code)
    starts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            starts.setdefault(node.lineno, []).append(node)
    separators = {}
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.OP and token.string == ';':
            separators.setdefault(token.start[0], []).append(token.start[1])
    out = []
    for number, line in enumerate(code.splitlines(), 1):
        nodes = starts.get(number, [])
        first = min(nodes, key=lambda n: n.col_offset) if nodes else None
        # A one-line if/for/function suite must stay inside its compound statement.
        simple = isinstance(first, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                                    ast.Expr, ast.Return, ast.Raise, ast.Assert,
                                    ast.Import, ast.ImportFrom))
        if number in separators and simple:
            indent = line[:len(line)-len(line.lstrip())]
            start = 0
            for end in separators[number] + [len(line)]:
                part = line[start:end].strip()
                if part:
                    out.append(indent + part)
                start = end + 1
        else:
            out.append(line)
    result = '\n'.join(out)
    assert ast.dump(ast.parse(result)) == ast.dump(tree), 'Formatting changed program semantics'
    return result


def comments_above(code):
    """Move trailing comments above their code line, preserving program syntax."""
    tree = ast.parse(code)
    lines = code.splitlines()
    comments = {}
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            number, column = token.start
            if lines[number-1][:column].strip():
                comments[number] = (column, token.string)
    out = []
    for number, line in enumerate(lines, 1):
        if number in comments:
            column, comment = comments[number]
            indent = line[:len(line)-len(line.lstrip())]
            out.append(indent + comment)
            line = line[:column].rstrip()
        out.append(line)
    result = '\n'.join(out)
    assert ast.dump(ast.parse(result)) == ast.dump(tree), 'Moving comments changed program semantics'
    return result


def annotate_python(code):
    code = comments_above(separate_statements(code))
    tree = ast.parse(code)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = {}
    seen = set()
    for node in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)),
                       key=lambda n: (n.lineno, n.col_offset)):
        name = ast.unparse(node.func)
        key = name if name in OPERATIONS else (
            node.func.attr if isinstance(node.func, ast.Attribute) and node.func.attr in METHODS else None)
        if not key or key in seen:
            continue
        statement = node
        while not isinstance(statement, ast.stmt):
            statement = parents[statement]
        # A return value may have a multiline expression: explain it before the
        # statement, not inside an argument list or an existing expression.
        text = OPERATIONS.get(key) or METHODS[key]
        found.setdefault(statement.lineno, []).append(text)
        seen.add(key)
    lines = code.splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        comments = found.get(i, [])
        # Existing comments immediately above the operation take precedence.
        if comments and not (i > 1 and lines[i-2].lstrip().startswith('#')):
            indent = line[:len(line)-len(line.lstrip())]
            for comment in comments[:2]:
                out.extend(indent+'# '+part for part in comment.splitlines())
        out.append(line)
    result = wrap_comments('\n'.join(out))
    assert ast.dump(ast.parse(result)) == ast.dump(tree), 'Comments changed program semantics'
    return result


RUST_OPERATIONS = {
    'autodiff()': 'Включаем запись вычислений, чтобы затем получить градиенты ошибки.',
    'LinearConfig::new': 'Задаём размеры линейного слоя; init создаёт его обучаемые веса.',
    'Conv2dConfig::new': 'Задаём свёртку: число входных и выходных каналов и размер фильтра.',
    'CrossEntropyLossConfig::new': 'Ошибка классификации сравнивает оценки классов с верными метками.',
    'Param::from_tensor': 'Регистрируем тензор как обучаемый параметр модели.',
    'AdamConfig::new': 'Adam подбирает шаг обновления для каждого параметра по истории градиентов.',
    'GradientsParams::from_grads': 'Связываем градиенты вычислительного графа с параметрами этой модели.',
    '.backward()': 'Обратный проход вычисляет градиенты ошибки.',
    'optimizer.step': 'Обновляем параметры модели; первый аргумент задаёт скорость обучения.',
    '.matmul(': 'Матрично умножаем тензоры: внутренние размеры должны совпадать.',
    '.detach()': 'Отделяем тензор от графа градиентов, сохраняя его значения.',
    '.require_grad()': 'Просим сохранять градиент по этому тензору.',
    '.slice(': 'Берём часть тензора по указанным диапазонам индексов.',
    '.reshape(': 'Меняем форму тензора, сохраняя число и порядок элементов.',
    'softmax(': 'Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.',
    '.argmax(': 'Выбираем индекс наибольшей оценки — предсказанную категорию.',
    '.into_scalar': 'Читаем единственное значение тензора как обычное число Rust.',
    '.svd(': 'Разлагаем матрицу на сингулярные направления и их значимость.',
    '.solve(': 'Решаем систему линейных уравнений без явного обращения матрицы.',
}


def annotate_rust(code):
    out, seen = [], set()
    for line in code.splitlines():
        if not line.lstrip().startswith('//'):
            matches = [(key, text) for key, text in RUST_OPERATIONS.items() if key in line and key not in seen]
            if matches and not (out and out[-1].lstrip().startswith('//')):
                indent = line[:len(line)-len(line.lstrip())]
                out.append(indent+'// '+matches[0][1])
                seen.add(matches[0][0])
        out.append(line)
    return wrap_comments('\n'.join(out), '//')

def wrap_comments(code, marker='#', width=88):
    lines = []
    for line in code.splitlines():
        match = re.match(r'^(\s*)' + re.escape(marker) + r' ?(.*)$', line)
        # A comment introduces a new explained step. Separate it from the
        # preceding calculation, while keeping comment paragraphs together.
        if match and lines and lines[-1].strip():
            previous = lines[-1].strip()
            if not previous.startswith(marker) and not previous.endswith((':', '{', '(')):
                lines.append('')
        if match and len(line) > width:
            prefix = match[1] + marker + ' '
            lines.extend(textwrap.wrap(match[2], width=width,
                                       initial_indent=prefix, subsequent_indent=prefix,
                                       break_long_words=False, break_on_hyphens=False))
        else:
            lines.append(line)
    return '\n'.join(lines).strip() + '\n'


def listing(imports, body, runtimes):
    """Keep the experiment body once; the reader adds the selected runtime."""
    body = annotate_python(body)
    used = {node.id for node in ast.walk(ast.parse(imports + '\n' + body +
            '\n'.join(runtimes.values()))) if isinstance(node, ast.Name)}
    lines = imports.splitlines()
    for node in ast.parse(imports).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if all((alias.asname or alias.name.split('.')[0]) not in used for alias in node.names):
                for i in range(node.lineno - 1, node.end_lineno):
                    lines[i] = ''
    prefix = '\n'.join(line for line in lines if line).strip()
    for runtime in runtimes.values():
        ast.parse(prefix + '\n\n' + runtime + '\n' + body)
    return {'prefix': prefix, 'body': body}


TORCH_RUNTIMES = {}
for device in ('cpu', 'cuda', 'mps'):
    check = {
        'cpu': '',
        'cuda': 'if not torch.cuda.is_available():\n'
                '    raise RuntimeError("CUDA недоступна: проверь видеокарту NVIDIA и установку PyTorch.")\n\n',
        'mps': 'if not torch.backends.mps.is_available():\n'
               '    raise RuntimeError("MPS недоступен: нужен совместимый Mac и нативный запуск.")\n\n',
    }[device]
    TORCH_RUNTIMES[device] = (check + f'device = torch.device("{device}")\n'
        '# Новые тензоры и параметры модели создаются на выбранном устройстве.\n'
        'torch.set_default_device(device)\n'
        '# float32 — вещественные числа по 32 бита на значение.\n'
        'torch.set_default_dtype(torch.float32)\n'
        '# Начальное состояние генератора позволяет повторить случайные данные.\n'
        'torch.manual_seed(7)\n' + ('torch.set_num_threads(1)\n' if device == 'cpu' else '') +
        'print("PyTorch", torch.__version__, "| устройство:", device)\n')

MLX_RUNTIMES = {
    'cpu': 'mx.set_default_device(mx.cpu)\n'
           '# Начальное состояние генератора позволяет повторить случайные данные.\n'
           'mx.random.seed(7)\nprint("MLX | устройство: CPU")\n',
    'metal': 'if mx.device_count(mx.gpu) == 0:\n'
             '    raise RuntimeError("Metal недоступен: нужен нативный запуск на Apple silicon.")\n\n'
             'mx.set_default_device(mx.gpu)\n'
             '# Начальное состояние генератора позволяет повторить случайные данные.\n'
             'mx.random.seed(7)\nprint("MLX | устройство: Metal")\n',
}

# Keep the setup as readable as the actual experiment.
TORCH_RUNTIMES = {device: wrap_comments(code) for device, code in TORCH_RUNTIMES.items()}
MLX_RUNTIMES = {device: wrap_comments(code) for device, code in MLX_RUNTIMES.items()}
