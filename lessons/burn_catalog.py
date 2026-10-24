"""Independent Burn programs with the same learning task as the PyTorch tab."""
from textwrap import dedent
from catalog import LESSONS

RUST = {}
EXERCISES = {
    'transformer': 'Функция attention вычисляет внимание: взвешивает сведения других позиций '
                   'по их сходству с запросом. В двух вызовах, создающих answer и changed, '
                   'замени последний аргумент true на false. Он задаёт запрет на будущие '
                   'позиции. Теперь запрет снят: проверь, меняются ли ранние ответы '
                   'после изменения последнего элемента.',
    'optimization': 'Здесь обновление параметра написано вручную. Добавь перед внутренним '
                    'циклом нулевой тензор velocity той же формы, что parameter. '
                    'На каждом шаге умножай прежний velocity на 0.9 и прибавляй текущий '
                    'градиент. Вместо самого градиента вычитай из parameter произведение '
                    'velocity на rate. Переменная velocity хранит накопленное направление, '
                    'а rate задаёт размер шага. Сравни уменьшение ошибки при тех же '
                    'начальных значениях.',
}
CARGO = '''[package]
name = "atlas-burn-example"
version = "0.1.0"
edition = "2024"
rust-version = "1.95"
publish = false

[features]
default = []
cuda = ["burn/cuda"]
metal = ["burn/metal"]

[dependencies]
burn = { version = "=0.22.1", default-features = false, features = ["std", "flex", "optim", "linalg"] }
'''
HEADER = '''use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
'''
def add(ids, code, explanation, expected, limits=None):
    for topic in ids.split():
        lesson = LESSONS[topic]
        header = HEADER if 'Distribution::' in code else HEADER.replace('Distribution, ', '')
        RUST[topic] = dict(goal=lesson['goal'], code=header+dedent(code).strip()+'\n',
            cargo=CARGO, explanation=explanation, expected=expected,
            exercise=EXERCISES.get(topic, lesson['exercise']), kind=lesson['kind'],
            limits=limits or lesson['limits'],
            parity='Та же учебная задача, размеры данных и разделение обучения/проверки, что в PyTorch. Генераторы случайных чисел различаются, поэтому значения не обязаны совпадать побитово.')

add('linear classification lab-table', '''
use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams}};
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([300, 2], Distribution::Normal(0.0, 1.0), &device);
    let y = x.clone().slice([0..300, 0..1])*3.0
        - x.clone().slice([0..300, 1..2])*2.0 + 0.7
        + Tensor::<2>::random([300, 1], Distribution::Normal(0.0, 0.1), &device);
    let mut model = LinearConfig::new(2, 1).init(&device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        let loss = (model.forward(x.clone().slice([0..240]))-y.clone().slice([0..240]))
            .powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let error = (model.forward(x.slice([240..300]))-y.slice([240..300]))
        .powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка на 60 новых строках: {error}");
    println!("Коэффициенты: {:?}; сдвиг: {:?}",
        model.weight.val().into_data().try_to_vec::<f32>().unwrap(),
        model.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap());
    assert!(error < 0.06, "Модель не выучила зависимость");
}
''', 'Burn обучает линейный слой на 240 строках из 300. Оптимизатор получает градиенты только от учебной части; последние 60 строк используются для оценки.',
    'Коэффициенты близки к 3 и −2, сдвиг — к 0.7. Ошибка новых строк мала, но не нулевая из-за шума.')

CLASSIFIER = '''
use burn::{module::Module, nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { hidden: Linear, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.output.forward(silu(self.hidden.forward(x)))
    }
}
fn accuracy(model: &Model, x: Tensor<2>, y: Tensor<1, Int>) -> f32 {
    model.forward(x).argmax(1).squeeze::<1>().equal(y).float().mean().into_scalar()
}
fn main() {
    let device = device().autodiff();
    __DATA__
    let mut model = Model { hidden: LinearConfig::new(2, __HIDDEN__).init(&device),
        output: LinearConfig::new(__HIDDEN__, 2).init(&device) };
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamConfig::new().init();
    let before = accuracy(&model, x.clone().slice([__TRAIN__..__COUNT__]), y.clone().slice([__TRAIN__..__COUNT__]));
    for _ in 0..160 {
        let loss = loss_fn.forward(model.forward(x.clone().slice([0..__TRAIN__])), y.clone().slice([0..__TRAIN__]));
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let after = accuracy(&model, x.slice([__TRAIN__..__COUNT__]), y.slice([__TRAIN__..__COUNT__]));
    println!("Доля верных ответов до: {before:.3}; после: {after:.3}");
    assert!(after > 0.9, "Недостаточная точность на новых точках");
}
'''
for ids, count, train, hidden, data in [
    ('mlp',500,400,24,'''let x = Tensor::<2>::random([500, 2], Distribution::Uniform(-1.0, 1.0), &device);
    let y = x.clone().powf_scalar(2.0).sum_dim(1).lower_elem(0.45).int().squeeze::<1>();'''),
    ('supervised scratch',320,240,12,'''let x = Tensor::<2>::random([320, 2], Distribution::Normal(0.0, 1.0), &device);
    let y = (x.clone().slice([0..320, 0..1])+x.clone().slice([0..320, 1..2])*0.5)
        .greater_elem(0.0).int().squeeze::<1>();'''),
]:
    code=CLASSIFIER.replace('__DATA__',data)
    for key,value in [('COUNT',count),('TRAIN',train),('HIDDEN',hidden)]:
        code=code.replace('__'+key+'__',str(value))
    add(ids,code,'Два линейных слоя учатся различать две категории. Между слоями функция SiLU нелинейно преобразует значения, чтобы сеть могла строить сложную границу. Функция ошибки CrossEntropy сравнивает оценки модели с правильной категорией. Последние объекты не участвуют в обновлении весов.',
        'Доля правильных ответов на новых точках выше 0.9.')

add('cnn vision', '''
use burn::{module::Module, nn::{Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{TensorData, Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { conv: Conv2d, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let n = x.dims()[0];
        let features = silu(self.conv.forward(x)).mean_dim(2).mean_dim(3).reshape([n, 8]);
        self.output.forward(features)
    }
}
fn main() {
    let device = device().autodiff();
    // 240 картинок, два направления полос, пять возможных сдвигов.
    let labels: Vec<i32> = (0..240).map(|i| i % 2).collect();
    let mut pixels = vec![0.0_f32; 240*64];
    for i in 0..240 {
        let position = 1 + (i/2)%5;
        for row in 0..8 { for col in 0..8 {
            let axis = if labels[i]==0 { col } else { row };
            if axis >= position && axis < position+2 { pixels[i*64+row*8+col]=1.0; }
        }}
    }
    let x = Tensor::<4>::from_data(TensorData::new(pixels, [240, 1, 8, 8]), &device)
        + Tensor::<4>::random([240, 1, 8, 8], Distribution::Normal(0.0, 0.12), &device);
    let y = Tensor::<1, Int>::from_data(TensorData::new(labels, [240]), &device);
    let mut model = Model {
        conv: Conv2dConfig::new([1, 8], [3, 3]).with_padding(PaddingConfig2d::Same).init(&device),
        output: LinearConfig::new(8, 2).init(&device) };
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..160 {
        let loss = loss_fn.forward(model.forward(x.clone().slice([0..180])), y.clone().slice([0..180]));
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let accuracy = model.forward(x.slice([180..240])).argmax(1).squeeze::<1>()
        .equal(y.slice([180..240])).float().mean().into_scalar::<f32>();
    println!("Верно на 60 новых картинках: {accuracy}; обучаемых чисел: {}", model.num_params());
    assert!(accuracy > 0.9, "CNN не научилась различать полосы");
}
''','Свёртка выделяет признаки, функция SiLU нелинейно преобразует их, усреднение объединяет сведения по изображению, а линейный слой выдаёт оценки категорий. Обучаем его на 180 картинках, затем проверяем 60 новых с другим шумом.',
    'На новых картинках точность выше 0.9. Сеть содержит 98 обучаемых чисел.')

add('lora', '''
use burn::{module::{Module, Param}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Adapter { a: Param<Tensor<2>>, b: Param<Tensor<2>> }
impl Adapter {
    fn forward(&self, x: Tensor<2>, base: Tensor<2>) -> Tensor<2> {
        x.clone().matmul(base) + x.matmul(self.a.val()).matmul(self.b.val())
    }
}
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([300, 8], Distribution::Normal(0.0, 1.0), &device);
    // Базовая матрица не входит в Adapter и не передаётся оптимизатору.
    let base = Tensor::<2>::random([8, 8], Distribution::Normal(0.0, 0.2), &device).detach();
    let base_before = base.clone();
    let true_update = Tensor::<2>::random([8, 2], Distribution::Normal(0.0, 1.0), &device)
        .matmul(Tensor::<2>::random([2, 8], Distribution::Normal(0.0, 1.0), &device))*0.1;
    let y = x.clone().matmul(base.clone()+true_update).detach();
    let mut adapter = Adapter {
        a: Param::from_tensor(Tensor::<2>::random([8, 2], Distribution::Normal(0.0, 0.01), &device)),
        b: Param::from_tensor(Tensor::<2>::zeros([2, 8], &device)) };
    let a_before = adapter.a.val().detach();
    let mut optimizer = AdamConfig::new().init();
    let test_x = x.clone().slice([240..300]);
    let test_y = y.clone().slice([240..300]);
    let before = (adapter.forward(test_x.clone(), base.clone())-test_y.clone())
        .powf_scalar(2.0).mean().into_scalar::<f32>();
    for _ in 0..250 {
        let prediction = adapter.forward(x.clone().slice([0..240]), base.clone());
        let loss = (prediction-y.clone().slice([0..240])).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &adapter);
        adapter = optimizer.step(0.04, adapter, grads);
    }
    let prediction = adapter.forward(test_x.clone(), base.clone());
    let after = (prediction.clone()-test_y).powf_scalar(2.0).mean().into_scalar::<f32>();
    let merged = base.clone()+adapter.a.val().matmul(adapter.b.val());
    let merge_error = (prediction-test_x.matmul(merged)).abs().max().into_scalar::<f32>();
    let base_change = (base-base_before).abs().max().into_scalar::<f32>();
    let a_change = (adapter.a.val()-a_before).abs().max().into_scalar::<f32>();
    let b_change = adapter.b.val().abs().max().into_scalar::<f32>();
    println!("MSE на 60 новых объектах: до {before:.6}, после {after:.6}");
    println!("Базовых весов: 64; обучаемых: {}; изменение базы: {base_change}", adapter.num_params());
    println!("Ошибка слияния адаптера: {merge_error}");
    assert!(after < before*0.02 && after < 0.001);
    assert_eq!(base_change, 0.0);
    assert!(a_change > 0.0001 && b_change > 0.0001);
    assert!(merge_error < 0.0001);
}
''','На 240 объектах обучаются обе матрицы адаптера A и B; базовые 64 веса неизменны. На 60 новых объектах измеряем ошибку до и после обучения. Дополнительно проверяем, что адаптер можно слить с базовой матрицей без изменения ответа.',
    'Ошибка новых объектов падает минимум в 50 раз. Обучаются 32 числа вместо 64; база не меняется. Слитая и раздельная модели дают одинаковые ответы с учётом округления.')

add('ae', '''
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Autoencoder { encode: Linear, decode: Linear }
impl Autoencoder {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.decode.forward(self.encode.forward(x)) }
}
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([300, 1], Distribution::Normal(0.0, 1.0), &device)
        .matmul(Tensor::<2>::from_data([[1.0_f32, 2., -1., 0.5]], &device))
        + Tensor::<2>::random([300, 4], Distribution::Normal(0.0, 0.03), &device);
    let mut model = Autoencoder { encode: LinearConfig::new(4, 1).init(&device),
        decode: LinearConfig::new(1, 4).init(&device) };
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        let train = x.clone().slice([0..240]);
        let loss = (model.forward(train.clone())-train).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let test = x.slice([240..300]);
    let ordinary = (model.forward(test.clone())-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    let strange = Tensor::<2>::from_data([[5.0_f32, -5., 5., -5.]], &device);
    let unusual = (model.forward(strange.clone())-strange).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Обычные новые объекты: {ordinary}; необычный: {unusual}");
    assert!(ordinary < 0.02 && unusual > ordinary*100.0);
}
''','Автоэнкодер сжимает четыре связанных числа в одно. После обучения сравниваем восстановление новых обычных объектов и объекта, нарушающего исходную связь.',
    'Ошибка обычных объектов мала; ошибка необычного как минимум в 100 раз больше.')

add('reservoir', '''
use burn::linalg::svd;
fn main() {
    let device = device();
    let w = Tensor::<2>::random([24, 24], Distribution::Normal(0.0, 0.08), &device);
    let input = Tensor::<2>::random([24, 1], Distribution::Normal(0.0, 0.3), &device);
    let mut state = Tensor::<2>::zeros([24, 1], &device);
    let mut states = Vec::new();
    for t in 0..259 {
        state = (w.clone().matmul(state)+input.clone()*(t as f64*0.13).sin()).tanh();
        states.push(state.clone().transpose());
    }
    let x = Tensor::cat(states, 0);
    let target: Vec<f32> = (1..260).map(|t| (t as f32*0.13).sin()).collect();
    let y = Tensor::<2>::from_data(burn::tensor::TensorData::new(target, [259, 1]), &device);
    // Подбираем выходные веса со штрафом за их величину.
    // Сингулярное разложение даёт то же решение, что (XᵀX + λI)⁻¹Xᵀy.
    let (u, s, vt) = svd::<2, 1>(x.clone().slice([20..200]), 100);
    let shrink = s.clone()/(s.powf_scalar(2.0)+0.001);
    let readout = (vt.transpose()*shrink.unsqueeze_dim::<2>(0))
        .matmul(u.transpose()).matmul(y.clone().slice([20..200]));
    let error = (x.slice([200..259]).matmul(readout)-y.slice([200..259]))
        .powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка прогноза на более позднем участке: {error}");
    assert!(error < 0.02);
}
''','Случайная рекуррентная сеть из 24 ячеек остаётся неизменной. Обучаем только выход на отсчётах 20–199, подбирая веса со штрафом за их величину через сингулярное разложение (SVD), затем проверяем более поздние отсчёты.',
    'Ошибка прогноза следующего значения синуса меньше 0.02.',
    'Сеть не обучает внутренние веса. Сингулярное разложение (SVD) в Burn 0.22.1 использует реализацию на CPU даже при GPU-устройстве; это ограничение ускорения, а не замена задачи.')

add('factorweights', '''
use burn::linalg::svd;
fn main() {
    let device = device();
    let weights = Tensor::<2>::random([64, 4], Distribution::Normal(0.0, 1.0), &device)
        .matmul(Tensor::<2>::random([4, 32], Distribution::Normal(0.0, 1.0), &device))
        + Tensor::<2>::random([64, 32], Distribution::Normal(0.0, 0.01), &device);
    let (u, s, vt) = svd::<2, 1>(weights.clone(), 100);
    let left = u.slice([0..64, 0..4])*s.slice([0..4]).unsqueeze_dim::<2>(0);
    let right = vt.slice([0..4, 0..32]);
    let x = Tensor::<2>::random([20, 32], Distribution::Normal(0.0, 1.0), &device);
    let error = (x.clone().matmul(right.transpose()).matmul(left.transpose())
        - x.matmul(weights.transpose())).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка: {error}; параметров до: 2048, после: 384");
    assert!(error < 0.02);
}
''','Вычисляем сингулярное разложение (SVD) исходной матрицы и оставляем четыре главных направления. Сравниваем полную и сжатую модели на новых входах.',
    'Количество весов падает с 2048 до 384 при малой ошибке.',
    'Сингулярное разложение (SVD) в Burn 0.22.1 выполняет основную декомпозицию на CPU. Умножения можно выполнять на выбранном GPU. Здесь нет обучения адаптера LoRA: это отдельная задача сжатия уже заданных весов.')

add('arrays', '''
fn main() {
    let device = device();
    let x = Tensor::<2>::from_data([[1.0_f32,2.,3.],[4.,5.,6.]], &device);
    let w = Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[1.,1.]], &device);
    let answer = x.matmul(w);
    println!("Форма: {:?}; ответ: {:?}", answer.dims(), answer.clone().into_data());
    assert_eq!(answer.into_data().try_to_vec::<f32>().unwrap(), vec![4.,5.,10.,11.]);
}
''','Размеры внутренних осей совпадают: [2,3] × [3,2] = [2,2].','Ответ [[4,5],[10,11]].')

add('derivatives', '''
fn main() {
    let device = device().autodiff();
    let x = Tensor::<1>::from_data([3.0_f32], &device).require_grad();
    let y = x.clone().powf_scalar(2.0)+x.clone()*2.0;
    let grads = y.clone().backward();
    let derivative = x.grad(&grads).unwrap().into_scalar::<f32>();
    let h = 0.001;
    let moved = x.detach()+h;
    let numerical = ((moved.clone().powf_scalar(2.0)+moved*2.0-y.detach())/h).into_scalar::<f32>();
    println!("Производная Burn: {derivative}; численная оценка: {numerical}");
    assert!((derivative-8.0).abs()<1e-6 && (numerical-8.0).abs()<0.01);
}
''','Сравниваем автоматическое дифференцирование с малым изменением входа.','Оба результата близки к 8.')

add('optimization', '''
fn main() {
    let device = device().autodiff();
    for rate in [0.05, 0.4, 1.1] {
        let mut parameter = Tensor::<1>::from_data([5.0_f32], &device).require_grad();
        let mut losses = Vec::new();
        for _ in 0..20 {
            let loss = (parameter.clone()-2.0).powf_scalar(2.0).mean();
            losses.push(loss.clone().into_scalar::<f32>());
            let grads = loss.backward();
            parameter = (parameter.clone()-parameter.grad(&grads).unwrap()*rate).detach().require_grad();
        }
        println!("Шаг: {rate}; первые ошибки: {:?}; последняя: {}", &losses[..3], losses[19]);
        assert!(if rate<1.0 { losses[19]<losses[0] } else { losses[19]>losses[0] });
    }
}
''','Применяем градиентный спуск (SGD): параметр минус скорость обучения, умноженная на градиент. Сравниваем те же три скорости и 20 шагов.','Малые шаги уменьшают ошибку, шаг 1.1 приводит к расходимости.')

add('compactarch specialblocks tinyml', '''
use burn::{module::Module, nn::{conv::Conv2dConfig, PaddingConfig2d}};
fn main() {
    let device = device();
    let regular = Conv2dConfig::new([16,32],[3,3]).with_padding(PaddingConfig2d::Same).with_bias(false).init(&device);
    let depthwise = Conv2dConfig::new([16,16],[3,3]).with_groups(16).with_padding(PaddingConfig2d::Same).with_bias(false).init(&device);
    let pointwise = Conv2dConfig::new([16,32],[1,1]).with_bias(false).init(&device);
    let x = Tensor::<4>::random([1,16,24,24], Distribution::Normal(0.0,1.0), &device);
    let full = regular.forward(x.clone());
    let compact = pointwise.forward(depthwise.forward(x));
    let a=regular.num_params(); let b=depthwise.num_params()+pointwise.num_params();
    println!("Обычная: {a} весов, {} байт, {:?}", a*4, full.dims());
    println!("Компактная: {b} весов, {} байт, {:?}", b*4, compact.dims());
    assert_eq!((a,b),(4608,656)); assert_eq!(full.dims(),compact.dims());
}
''','Создаём оба варианта блока, считаем их реальные параметры, память весов в формате FP32 — 32-битных вещественных чисел — и форму ответа.','4608 и 656 весов; одинаковые формы [1,32,24,24].')

add('ssm', '''
fn main() {
    let device = device();
    let decay = Tensor::<1>::from_data([0.1_f32,0.5,0.8,0.95], &device);
    let mut state = Tensor::<1>::zeros([4], &device);
    let mut outputs=Vec::new();
    for t in 0..30 {
        state=decay.clone()*state+if t==3 {1.0} else {0.0};
        outputs.push(state.clone().mean().into_scalar::<f32>());
    }
    println!("Ответ после импульса: {:?}", &outputs[3..12]);
    assert!(outputs[3]>outputs[8] && outputs[8]>outputs[20] && outputs[20]>0.0);
}
''','Четыре коэффициента затухания создают четыре масштаба памяти; выход — среднее состояний.','Память после импульса затухает с разными скоростями.')

ATTENTION = '''
use burn::tensor::{TensorData, Bool, activation::softmax};
fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, causal: bool) -> Tensor<4> {
    let [batch, heads, length, width]=q.dims();
    let key_length=k.dims()[2];
    let mut scores=q.matmul(k.swap_dims(2,3))/(width as f64).sqrt();
    if causal {
        let mask: Vec<bool>=(0..length).flat_map(|i|(0..key_length).map(move|j|j>i)).collect();
        let mask=Tensor::<4,Bool>::from_data(TensorData::new(mask,[1,1,length,key_length]), &scores.device());
        scores=scores.mask_fill(mask.expand([batch,heads,length,key_length]), -1e9);
    }
    softmax(scores,3).matmul(v)
}
'''
add('transformer', ATTENTION+'''
use burn::nn::LinearConfig;
fn main() {
    let device = device();
    let x=Tensor::<3>::random([2,6,16],Distribution::Normal(0.0,1.0),&device);
    let projection=LinearConfig::new(16,48).with_bias(false).init(&device).forward(x);
    let parts: Vec<Tensor<4>>=(0..3).map(|i|projection.clone().slice([0..2,0..6,i*16..(i+1)*16]).reshape([2,6,4,4]).swap_dims(1,2)).collect();
    let answer=attention(parts[0].clone(),parts[1].clone(),parts[2].clone(),true);
    let v=parts[2].clone();
    let changed=v.clone().slice_assign([0..2,0..4,5..6,0..4],v.slice([0..2,0..4,5..6,0..4])+100.0);
    let changed=attention(parts[0].clone(),parts[1].clone(),changed,true);
    let error=(answer.clone().slice([0..2,0..4,0..5])-changed.slice([0..2,0..4,0..5])).abs().max().into_scalar::<f32>();
    println!("Размер ответа: {:?}; изменение ранних ответов: {error}",answer.dims());
    assert_eq!(answer.dims(),[2,4,6,4]); assert!(error<1e-5);
}
''','Вычисляем четыре головы внимания с причинной маской, затем меняем последнее значение. Первые пять позиций не должны увидеть это изменение.',
    'Форма [2,4,6,4]; предыдущие позиции не меняются.')

add('cache', ATTENTION+'''
fn main() {
    let device = device();
    let x=Tensor::<4>::random([1,1,10,8],Distribution::Normal(0.0,1.0),&device);
    let q=x.clone()+0.1; let k=x.clone()-0.1; let v=x*0.5;
    let full=attention(q.clone(),k.clone(),v.clone(),true);
    let mut keys=Vec::new(); let mut values=Vec::new(); let mut outputs=Vec::new();
    for t in 0..10 {
        keys.push(k.clone().slice([0..1,0..1,t..t+1]));
        values.push(v.clone().slice([0..1,0..1,t..t+1]));
        outputs.push(attention(q.clone().slice([0..1,0..1,t..t+1]),Tensor::cat(keys.clone(),2),Tensor::cat(values.clone(),2),false));
    }
    let cached=Tensor::cat(outputs,2);
    let error=(full-cached).abs().max().into_scalar::<f32>();
    println!("Разница полного и пошагового расчёта: {error}"); assert!(error<1e-5);
}
''','Накапливаем ключи и значения, рассчитывая по одному запросу. Сравниваем весь результат с полным расчётом внимания, в котором каждая позиция видит только себя и прошлое.','Полный и пошаговый расчёты совпадают с учётом округления.')

add('gnn graphsdata', '''
use burn::nn::LinearConfig;
fn main() {
    let device = device();
    let adjacency=Tensor::<2>::from_data([[1.0_f32,1.,0.,0.],[1.,1.,1.,0.],[0.,1.,1.,0.],[0.,0.,0.,1.]],&device);
    let features=Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[1.,1.],[8.,8.]],&device);
    let weights=adjacency.clone()/adjacency.sum_dim(1);
    let messages=weights.matmul(features.clone());
    let answer=LinearConfig::new(2,3).with_bias(false).init(&device).forward(messages.clone());
    println!("Сведения после обмена: {:?}", messages.clone().into_data());
    println!("Размер после слоя: {:?}",answer.dims());
    let error=(messages.slice([3..4])-features.slice([3..4])).abs().max().into_scalar::<f32>();
    assert_eq!(answer.dims(),[4,3]); assert_eq!(error,0.0);
}
''','Те же четыре объекта, связи и признаки. Усредняем соседей и применяем линейный слой; изолированный объект сохраняет свои исходные сведения.','Последняя строка сообщений равна [8,8], форма результата [4,3].')

add('knn', '''
fn main() {
    let device=device();
    let x=Tensor::<2>::random([240,2],Distribution::Normal(0.0,1.0),&device);
    let y=(x.clone().slice([0..240,0..1])+x.clone().slice([0..240,1..2])*0.5).greater_elem(0.0).int();
    let test=x.clone().slice([180..240]).reshape([60,1,2]);
    let train=x.slice([0..180]).reshape([1,180,2]);
    let distance=(test-train).powf_scalar(2.0).sum_dim(2).squeeze_dim::<2>(2);
    let (_,indices)=(-distance).topk_with_indices(3,1);
    let labels=y.clone().slice([0..180]).transpose().expand([60,180]).gather(1,indices.clone());
    let prediction=labels.float().mean_dim(1).greater_equal_elem(0.5).int();
    let accuracy=prediction.equal(y.slice([180..240])).float().mean().into_scalar::<f32>();
    println!("Верно на новых точках: {accuracy}; первые соседи: {:?}",indices.slice([0..1]).into_data());
    assert!(accuracy>0.8);
}
''','Из 180 известных объектов выбираем три ближайших для каждого из 60 новых. Категория определяется большинством голосов.','Точность выше 0.8; показаны три соседа первого проверочного объекта.')

add('probability', '''
fn main() {
    let device=device();
    let observed=Tensor::<1>::from_data([1.0_f32,1.,0.,1.,1.,0.,1.,1.],&device);
    let a=1+observed.sum().into_scalar::<f32>() as usize;
    let b=1+8-(a-1);
    // Для целых параметров гамма-распределение получаем как сумму -ln(U).
    // U — случайное число от 0 до 1.
    // X/(X+Y), где X и Y из гамма-распределений, даёт бета-распределение.
    let ga=-Tensor::<2>::random([10000,a],Distribution::Uniform(0.0000001,1.0),&device).log().sum_dim(1);
    let gb=-Tensor::<2>::random([10000,b],Distribution::Uniform(0.0000001,1.0),&device).log().sum_dim(1);
    let samples=(ga.clone()/(ga+gb)).squeeze::<1>().sort(0);
    let low=samples.clone().slice([249..250]).into_scalar::<f32>();
    let high=samples.slice([9749..9750]).into_scalar::<f32>();
    let mean=a as f32/(a+b) as f32;
    println!("Оценка вероятности: {mean}; 95% интервал: [{low}, {high}]");
    assert!((mean-0.7).abs()<1e-6 && low<0.7 && high>0.7 && low>0.3 && high<0.99);
}
''','Бета-распределение описывает возможные значения вероятности от 0 до 1. Начинаем с параметров 1 и 1; шесть успехов и две неудачи увеличивают их до 7 и 3. Из этого распределения генерируем 10 000 значений и находим границы, за которыми лежит по 2.5% выборки.',
    'Средняя вероятность 0.7; интервал примерно [0.40,0.93], с небольшим случайным отклонением.',
    'Способ генерации через сумму логарифмов использует целые параметры бета-распределения; для произвольных вещественных параметров нужен общий генератор гамма-распределения. Интервал описывает неопределённость вероятности, а не гарантирует исход следующего опыта.')
