"""Burn 0.22 API: Device and Tensor<D>, without a backend type parameter."""
from textwrap import dedent
RUST = {}
CARGO = '''[package]
name = "atlas-burn-example"
version = "0.1.0"
edition = "2024"
rust-version = "1.95"
publish = false

[dependencies]
burn = { version = "=0.22.1", default-features = false, features = ["std", "flex", "optim"] }
'''
def add(ids, goal, code, explanation, expected, limits='Это маленький учебный опыт на CPU. Результат не заменяет проверку на реальных данных и целевом устройстве.'):
    for id in ids.split():
        RUST[id] = dict(goal=goal, code=dedent(code).strip()+'\n', cargo=CARGO,
          explanation=explanation, expected=expected, limits=limits, kind='Самостоятельный пример на Rust')

add('linear classification supervised scratch lab-table', 'Обучить линейную модель правилу y = 3x + 2 и проверить новое значение x.', '''
use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor, TensorData}};
fn main() {
    // Flex считает на CPU. autodiff включает вычисление градиентов.
    let device = Device::flex().autodiff();
    let x = Tensor::<2>::from_data(TensorData::new(vec![-1.0_f32, 0., 1., 2.], [4, 1]), &device);
    let y = x.clone() * 3.0 + 2.0;
    let mut model = LinearConfig::new(1, 1).init(&device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        let loss = (model.forward(x.clone()) - y.clone()).powf_scalar(2.0).mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.05, model, gradients);
    }
    let test = Tensor::<2>::from_data([[3.0_f32]], &device);
    let prediction = model.forward(test).into_scalar::<f32>();
    println!("Для x=3 получено {prediction:.4}; ожидаем 11");
    assert!((prediction - 11.0).abs() < 0.1);
}
''', 'Создаём устройство с автоматическим дифференцированием, слой и оптимизатор. Каждый шаг вычисляет ошибку и градиенты, затем возвращает новую версию модели. В Burn 0.22 тип тензора содержит число измерений, а не тип backend.', 'Предсказание для x=3 близко к 11. Этот вход не использовался в обучении.')

add('arrays', 'Умножить пакет объектов на матрицу преобразования.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let x = Tensor::<2>::from_data([[1.0_f32, 2., 3.], [4., 5., 6.]], &device);
    let w = Tensor::<2>::from_data([[1.0_f32, 0.], [0., 1.], [1., 1.]], &device);
    let output = x.matmul(w);
    println!("Форма: {:?}", output.dims());
    let values = output.into_data().try_to_vec::<f32>().unwrap();
    println!("Значения: {values:?}");
    assert_eq!(values, vec![4., 5., 10., 11.]);
}
''', 'Tensor<2> — двумерный массив. Для произведения внутренние размеры должны совпасть: [2, 3] умножается на [3, 2].', 'Форма [2, 2], значения [4, 5, 10, 11].')

add('derivatives', 'Вычислить производную x² + 2x в точке 3.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex().autodiff();
    let x = Tensor::<1>::from_data([3.0_f32], &device).require_grad();
    let y = x.clone().powf_scalar(2.0) + x.clone()*2.0;
    let gradients = y.backward();
    let derivative = x.grad(&gradients).unwrap().into_scalar::<f32>();
    println!("Производная: {derivative}");
    assert!((derivative-8.0).abs() < 1e-5);
}
''', 'require_grad помечает вход, по которому нужна производная. backward проходит вычислительный граф назад; grad достаёт производную именно по x.', 'Производная равна 8.')

add('optimization', 'Обновлять один параметр вручную по градиенту.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex().autodiff();
    let mut value = Tensor::<1>::from_data([5.0_f32], &device).require_grad();
    for _ in 0..80 {
        let loss = (value.clone()-2.0).powf_scalar(2.0).mean();
        let gradients = loss.backward();
        let gradient = value.grad(&gradients).unwrap();
        // detach отделяет новый шаг от графа предыдущих шагов.
        value = (value-gradient*0.1).detach().require_grad();
    }
    let answer = value.into_scalar::<f32>();
    println!("Параметр после обучения: {answer}");
    assert!((answer-2.0).abs() < 1e-3);
}
''', 'Вычитаем производную, умноженную на размер шага. После каждого обновления начинаем новый вычислительный граф.', 'Параметр приближается к 2.')

add('mlp', 'Обучить двухслойную сеть приближать квадрат числа.', '''
use burn::{module::Module, nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{Device, Tensor, TensorData, activation::relu}};
#[derive(Module, Debug)]
struct Model { hidden: Linear, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.output.forward(relu(self.hidden.forward(x)))
    }
}
fn main() {
    let device = Device::flex().autodiff();
    let values: Vec<f32> = (0..80).map(|i| i as f32/40.0-1.0).collect();
    let x = Tensor::<2>::from_data(TensorData::new(values, [80, 1]), &device);
    let y = x.clone().powf_scalar(2.0);
    let mut model = Model { hidden: LinearConfig::new(1, 24).init(&device),
        output: LinearConfig::new(24, 1).init(&device) };
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..500 {
        let loss = (model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, gradients);
    }
    let test = Tensor::<2>::from_data([[-0.735_f32], [0.315], [0.825]], &device);
    let error = (model.forward(test.clone())-test.powf_scalar(2.0)).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка на новых числах: {error:.6}");
    assert!(error < 0.02);
}
''', 'Module собирает параметры обоих слоёв для оптимизатора. Между линейными слоями стоит ReLU: без нелинейности их композиция оставалась бы прямой.', 'Ошибка на трёх новых числах должна быть меньше 0.02.')

add('cnn vision', 'Проверить формы данных при свёртке изображения.', '''
use burn::{nn::{conv::Conv2dConfig, PaddingConfig2d}, tensor::{Device, Tensor}};
fn main() {
    let device = Device::flex();
    let image = Tensor::<4>::ones([2, 1, 8, 8], &device);
    let convolution = Conv2dConfig::new([1, 8], [3, 3])
        .with_padding(PaddingConfig2d::Same).init(&device);
    let features = convolution.forward(image);
    println!("После свёртки: {:?}", features.dims());
    assert_eq!(features.dims(), [2, 8, 8, 8]);
}
''', 'Четыре измерения — пакет, каналы, высота и ширина. Слой превращает один канал в восемь карт признаков; Same сохраняет размер изображения.', 'Форма [2, 8, 8, 8]: два изображения и восемь каналов.', 'Это прямой проход свёртки, без обучения классификатора. Полный учебный цикл для полос находится во вкладке Python.')

add('compactarch specialblocks tinyml', 'Собрать разделённую свёртку и посчитать хранимые веса.', '''
use burn::{nn::{conv::Conv2dConfig, PaddingConfig2d}, tensor::{Device, Tensor}};
fn main() {
    let device = Device::flex();
    let separate = Conv2dConfig::new([16, 16], [3, 3]).with_groups(16)
        .with_bias(false).with_padding(PaddingConfig2d::Same).init(&device);
    let mix = Conv2dConfig::new([16, 32], [1, 1]).with_bias(false).init(&device);
    let x = Tensor::<4>::ones([1, 16, 24, 24], &device);
    let result = mix.forward(separate.forward(x));
    let count = separate.weight.val().dims().iter().product::<usize>() + mix.weight.val().dims().iter().product::<usize>();
    println!("Форма: {:?}; весов: {count}", result.dims());
    assert_eq!(count, 656);
}
''', 'Первый слой обрабатывает каналы отдельно; второй смешивает их окнами 1 × 1. В конфигурации явно выключены дополнительные сдвиги.', '656 весов вместо 4608 у обычной свёртки с теми же каналами.', 'Показан блок, а не обученная сеть. Скорость и качество не следуют напрямую из числа весов.')

add('ssm reservoir', 'Пропустить импульс через затухающую память с четырьмя скоростями.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let decay = Tensor::<1>::from_data([0.1_f32, 0.5, 0.8, 0.95], &device);
    let mut state = Tensor::<1>::zeros([4], &device);
    for step in 0..15 {
        let input = if step == 3 { 1.0 } else { 0.0 };
        state = decay.clone()*state + input;
        println!("{step}: {:.4}", state.clone().mean().into_scalar::<f32>());
    }
}
''', 'На каждом шаге часть прошлого состояния сохраняется и добавляется новый вход. Коэффициенты около единицы дают более длинную память.', 'После третьего шага память плавно затухает.', 'Это линейная память с заданными коэффициентами, без обучения и без селективных блоков Mamba-3.')

add('factorweights lora', 'Показать, как низкоранговая добавка меняет выход без замены базовых весов.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let x = Tensor::<2>::ones([3, 8], &device);
    let base = Tensor::<2>::ones([8, 8], &device)*0.1;
    let a = Tensor::<2>::ones([8, 2], &device)*0.02;
    let b = Tensor::<2>::ones([2, 8], &device)*0.03;
    let separate = x.clone().matmul(base.clone()) + x.clone().matmul(a.clone()).matmul(b.clone());
    let merged = x.matmul(base+a.matmul(b));
    let error = (separate-merged).abs().max().into_scalar::<f32>();
    println!("Ошибка двух способов: {error}");
    assert!(error < 1e-5);
}
''', 'Произведение двух узких матриц можно вычислить отдельной веткой или заранее прибавить к базовым весам. Два пути должны совпадать.', 'Максимальная разница меньше 0.00001.', 'Это проверка алгебры добавки, не обучение LoRA. Обучение адаптера показано во вкладке Python.')

add('ae', 'Обучить линейный автоэнкодер с узким местом из одного числа.', '''
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor, TensorData}};
#[derive(Module, Debug)]
struct Autoencoder { encode: Linear, decode: Linear }
impl Autoencoder {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.decode.forward(self.encode.forward(x)) }
}
fn main() {
    let device = Device::flex().autodiff();
    let values: Vec<f32> = (0..60).flat_map(|i| { let t=i as f32/30.0-1.0; [t, 2.0*t, -t] }).collect();
    let x=Tensor::<2>::from_data(TensorData::new(values, [60, 3]), &device);
    let mut model=Autoencoder { encode: LinearConfig::new(3, 1).init(&device), decode: LinearConfig::new(1, 3).init(&device) };
    let mut optimizer=AdamConfig::new().init();
    for _ in 0..350 {
        let loss=(model.forward(x.clone())-x.clone()).powf_scalar(2.0).mean();
        let gradients=GradientsParams::from_grads(loss.backward(), &model);
        model=optimizer.step(0.03, model, gradients);
    }
    let test=Tensor::<2>::from_data([[0.315_f32, 0.63, -0.315]], &device);
    let error=(model.forward(test.clone())-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка восстановления нового объекта: {error}");
    assert!(error < 0.01);
}
''', 'Encoder сжимает три признака в один, decoder восстанавливает три. Сжатие возможно, потому что данные описывают одну скрытую величину.', 'Новый объект восстанавливается со средней квадратичной ошибкой меньше 0.01.')

add('transformer', 'Вычислить внимание: каждому запросу дать взвешенную сумму значений.', '''
use burn::tensor::{Device, Tensor, activation::softmax};
fn main() {
    let device = Device::flex();
    let query = Tensor::<2>::from_data([[1.0_f32, 0.], [0., 1.]], &device);
    let keys = Tensor::<2>::from_data([[1.0_f32, 0.], [0., 1.], [1., 1.]], &device);
    let values = Tensor::<2>::from_data([[10.0_f32], [20.], [30.]], &device);
    let scores = query.matmul(keys.transpose()) / (2.0_f64).sqrt();
    let attention = softmax(scores, 1);
    let answer = attention.clone().matmul(values);
    println!("Веса внимания: {:?}", attention.into_data().try_to_vec::<f32>().unwrap());
    println!("Ответы: {:?}", answer.into_data().try_to_vec::<f32>().unwrap());
}
''', 'Скалярное произведение запросов и ключей оценивает сходство. Softmax превращает оценки в положительные веса с суммой 1; затем они смешивают значения.', 'Два запроса дают два разных взвешенных ответа между 10 и 30.', 'Это прямой проход одной головы без причинной маски и обучения, не полный Transformer.')

add('cache', 'Проверить использование сохранённых ключей для очередного запроса.', '''
use burn::tensor::{Device, Tensor, activation::softmax};
fn main() {
    let device = Device::flex();
    let all_keys = Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[1.,1.]], &device);
    let all_values = Tensor::<2>::from_data([[2.0_f32],[4.],[8.]], &device);
    for length in 1..=3 {
        let keys = all_keys.clone().slice([0..length, 0..2]);
        let values = all_values.clone().slice([0..length, 0..1]);
        let query = Tensor::<2>::from_data([[1.0_f32,1.]], &device);
        let scores = softmax(query.matmul(keys.transpose()) / (2.0_f64).sqrt(), 1);
        let answer = scores.matmul(values).into_scalar::<f32>();
        println!("Доступно позиций: {length}; ответ: {answer:.4}");
    }
}
''', 'На каждом шаге доступны только текущий префикс ключей и значений. Предыдущие ключи не вычисляются заново.', 'При первой позиции ответ равен 2. По мере открытия префикса меняются доступные сведения и ответ.', 'Ключи заранее заданы; здесь нет динамического выделения кэша, генератора токенов и замера ускорения.')

add('gnn graphsdata', 'Передать сведения между соседями через матрицу связей.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let links = Tensor::<2>::from_data([[0.5_f32,0.5,0.],[0.5,0.5,0.],[0.,0.,1.]], &device);
    let features = Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[8.,8.]], &device);
    let shared = links.matmul(features).into_data().try_to_vec::<f32>().unwrap();
    println!("Признаки после обмена: {shared:?}");
    assert_eq!(shared, vec![0.5,0.5,0.5,0.5,8.,8.]);
}
''', 'Первые два объекта усредняют сведения друг друга. У третьего есть только связь с самим собой, поэтому его признаки остаются прежними.', 'Первые две строки становятся [0.5, 0.5], третья — [8, 8].', 'Это обмен сообщениями без обучения и без разреженного представления большого графа.')

add('knn', 'Найти ближайший объект по расстоянию в пространстве признаков.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let objects = Tensor::<2>::from_data([[0.0_f32,0.],[1.,1.],[3.,3.]], &device);
    let query = Tensor::<2>::from_data([[0.9_f32,1.1]], &device);
    let distances = (objects-query).powf_scalar(2.0).sum_dim(1);
    let values = distances.into_data().try_to_vec::<f32>().unwrap();
    let nearest = values.iter().enumerate().min_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
    println!("Квадраты расстояний: {values:?}; ближайший: {nearest}");
    assert_eq!(nearest, 1);
}
''', 'Broadcasting вычитает один запрос из каждой строки. Сумма квадратов по признакам даёт расстояние до каждого объекта; корень для выбора минимума не нужен.', 'Ближайший объект имеет индекс 1 — точка [1, 1].', 'Это точный поиск одного соседа на трёх объектах. Для больших коллекций необходим индекс и оценка качества поиска.')

add('probability', 'Обновить среднюю оценку вероятности после наблюдений.', '''
use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let observed = Tensor::<1>::from_data([1.0_f32,1.,0.,1.,1.,0.,1.,1.], &device);
    let success = observed.sum().into_scalar::<f32>();
    let a = 1.0 + success;
    let b = 1.0 + 8.0 - success;
    println!("Средняя вероятность после наблюдений: {}", a/(a+b));
    assert!((a/(a+b)-0.7).abs() < 1e-6);
}
''', 'Начинаем с Beta(1, 1). К первому параметру добавляем успехи, ко второму — неудачи; среднее нового распределения равно a/(a+b).', 'Шесть успехов и две неудачи дают оценку 0.7 с учётом начального распределения.', 'Пример вычисляет среднюю оценку. Для понимания неопределённости нужен и интервал, показанный во вкладке Python.')
