# Python 3.14 · MLX 0.32.4
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

mx.set_default_device(mx.cpu)

# Начальное состояние генератора позволяет повторить случайные данные.
mx.random.seed(7)
print("MLX | устройство: CPU")

def fit(model, x, y, steps=160, classification=False, rate=0.02):
    def loss_fn(model, x, y):
        prediction = model(x)
        if classification:
            # Измеряем ошибку классификации по исходным оценкам классов и верным меткам.
            return nn.losses.cross_entropy(prediction, y, reduction="mean")

        # Для численного прогноза сначала найдём ошибку каждого ответа.
        differences = prediction - y

        # Возводим каждый элемент в квадрат; так получают квадраты ошибок.
        squared_errors = mx.square(differences)

        # Усредняем значения вдоль указанной оси или по всему массиву.
        return mx.mean(squared_errors)

    # Вычисляем ошибку и градиенты по обучаемым параметрам модели MLX.
    value_and_grad = nn.value_and_grad(model, loss_fn)

    # Adam обновляет параметры, учитывая историю их градиентов.
    optimizer = optim.Adam(learning_rate=rate, bias_correction=True)
    for _ in range(steps):
        loss, grads = value_and_grad(model, x, y)

        # Применяем вычисленные градиенты к параметрам модели.
        optimizer.update(model, grads)

        # MLX откладывает расчёт до запроса результата.
        # Здесь выполняем накопленные вычисления весов, состояния и ошибки.
        mx.eval(model.parameters(), optimizer.state, loss)
    return loss.item()

# Создаём искусственные данные из нормального распределения со средним 0.
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
