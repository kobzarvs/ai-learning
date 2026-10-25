# Python 3.14 · MLX 0.32.4
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

def attention(q, k, v, causal=False):
    # Меняем две оси местами, чтобы согласовать формы перед матричным умножением.
    scores = (q@mx.swapaxes(k, -1, -2))/(q.shape[-1]**0.5)
    if causal:
        mask = mx.arange(k.shape[-2])[None, :] > mx.arange(q.shape[-2])[:, None]

        # По логической маске выбираем значение из одного из двух массивов.
        scores = mx.where(mask, -1e9, scores)

    # Преобразуем оценки в неотрицательные веса, сумма которых равна 1.
    return mx.softmax(scores, axis=-1)@v

# Создаём искусственные данные из нормального распределения со средним 0.
x = mx.random.normal((2, 6, 16))

# Линейный слой учится преобразованию y = x @ W.T + b.
projection = nn.Linear(16, 48, bias=False)

# Меняем форму массива, сохраняя число и порядок элементов.
# Разделяем массив на части вдоль выбранной оси.
q, k, v = [mx.swapaxes(z.reshape(2, 6, 4, 4), 1, 2)
           for z in mx.split(projection(x), 3, axis=-1)]
answer = attention(q, k, v, True)

# Соединяем массивы вдоль уже существующей оси.
changed_v = mx.concatenate([v[:, :, :-1], v[:, :, -1:]+100], axis=2)
changed = attention(q, k, changed_v, True)
print("Размер ответа:", answer.shape, "ранние ответы не меняются:", mx.allclose(answer[:, :, :-1], changed[:, :, :-1]).item())
assert mx.allclose(answer[:, :, :-1], changed[:, :, :-1]).item()
