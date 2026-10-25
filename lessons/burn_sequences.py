"""Sequence and representation tasks; standalone Burn 0.22.1 bodies."""

def register(add):
    add('rnn lstm', r'''
use burn::{module::Module, nn::{Lstm, LstmConfig, Linear, LinearConfig}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Model { memory: Lstm, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let n = x.dims()[0];
        let (states, _) = self.memory.forward(x, None);
        self.output.forward(states.slice([0..n, 11..12, 0..16]).reshape([n,16]))
    }
}
fn main() {
    let device = device().autodiff();
    let x = Tensor::<3>::random([300,12,1], Distribution::Normal(0.0,1.0), &device);
    let y = x.clone().mean_dim(1).reshape([300,1]).detach();
    let mut model = Model { memory: LstmConfig::new(1,16,true).init(&device), output: LinearConfig::new(16,1).init(&device) };
    let tx = x.clone().slice([240..300]); let ty = y.clone().slice([240..300]);
    let baseline = ty.clone().powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = (model.forward(tx.clone())-ty.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..260 {
        let loss = (model.forward(x.clone().slice([0..240]))-y.clone().slice([0..240])).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.01,model,grads);
    }
    let error = (model.forward(tx)-ty).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE до {before:.5}, после {error:.5}, нулевой прогноз {baseline:.5}");
    assert!(error < baseline*0.5 && error < before*0.5);
}
''', 'Lstm читает 12 чисел по очереди; output использует только последнее состояние. Adam обучает память и выход на 240 последовательностях, 60 отложены для проверки.', 'MSE на 60 новых последовательностях меньше половины ошибки нулевого прогноза и исходной сети.', exercise='Увеличь длину до 60: измени размеры x и срез последнего состояния на 59..60 в Model::forward. Затем замени импорты Lstm/LstmConfig на Rnn/RnnConfig, тип поля memory на Rnn и конструктор на RnnConfig::new(1,16,true).with_clip(None).init(&device); forward(x,None) также возвращает пару (states,state). В Burn 0.22.1 простая ячейка называется Rnn. Обучи обе модели на одних x и y: создай оба экземпляра после генерации данных и используй клоны тех же срезов 0..240 и 240..300. Сравни MSE при одинаковом числе шагов.')

    add('tcn audio lab-signal', r'''
use burn::{module::Module, nn::{conv::{Conv1d, Conv1dConfig}, Linear, LinearConfig, loss::CrossEntropyLossConfig}, optim::{AdamConfig, GradientsParams}, tensor::{Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { conv: Conv1d, output: Linear }
impl Model {
    fn forward(&self,x:Tensor<3>) -> Tensor<2> {
        self.output.forward(silu(self.conv.forward(x)).mean_dim(2).squeeze::<2>())
    }
}
fn accuracy(model:&Model,x:Tensor<3>,y:Tensor<1,Int>) -> f32 {
    model.forward(x).argmax(1).squeeze::<1>().equal(y).float().mean().into_scalar()
}
fn main() {
    let device=device().autodiff();
    let y=Tensor::<1>::random([300],Distribution::Uniform(0.0,1.0),&device).greater_elem(0.5).int();
    let frequency=y.clone().float().reshape([300,1,1])*6.0+3.0;
    let t=Tensor::<1>::from_floats((0..64).map(|i|i as f32/63.0).collect::<Vec<_>>().as_slice(),&device).reshape([1,1,64]);
    let x=(frequency*t*(2.0*std::f32::consts::PI)).sin()+Tensor::<3>::random([300,1,64],Distribution::Normal(0.0,0.15),&device);
    let mut model=Model{conv:Conv1dConfig::new(1,12,5).with_dilation(2).init(&device),output:LinearConfig::new(12,2).init(&device)};
    let mut optimizer=AdamConfig::new().init(); let loss_fn=CrossEntropyLossConfig::new().init(&device);
    let before=accuracy(&model,x.clone().slice([240..300]),y.clone().slice([240..300]));
    for _ in 0..180 {
        let loss=loss_fn.forward(model.forward(x.clone().slice([0..240])),y.clone().slice([0..240]));
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.02,model,grads);
    }
    let after=accuracy(&model,x.slice([240..300]),y.slice([240..300]));
    println!("Точность до {before:.3}, после {after:.3}"); assert!(after>0.9);
}
''', 'Conv1d с dilation=2 ищет рисунок колебаний 3 и 9 Гц. SiLU, усреднение по времени и Linear обучаются различать 240 шумных записей; последние 60 проверяют качество.', 'Точность на 60 новых сигналах превышает 0.9.', exercise='Добавь случайную фазу формы [300,1,1] к аргументу sin и заново обучи Model. Сравни точность.')

    add('siamese contrastive', r'''
use burn::{module::Module, nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct Encoder { hidden:Linear, output:Linear }
impl Encoder {
    fn forward(&self,x:Tensor<2>)->Tensor<2>{
        let z=self.output.forward(silu(self.hidden.forward(x)));
        let norm=(z.clone().powf_scalar(2.0).sum_dim(1)+1e-8).sqrt(); z/norm
    }
}
fn main(){
    let device=device().autodiff();
    let x=Tensor::<2>::random([128,8],Distribution::Normal(0.0,1.0),&device);
    let mut encoder=Encoder{hidden:LinearConfig::new(8,24).init(&device),output:LinearConfig::new(24,12).init(&device)};
    let labels=Tensor::<1,Int>::arange(0..128,&device);
    let loss_fn=CrossEntropyLossConfig::new().init(&device); let mut optimizer=AdamConfig::new().init();
    // Фиксированная обучающая пара для сопоставимого замера до/после.
    let probe_a=x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device);
    let probe_b=x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device);
    let before=loss_fn.forward(encoder.forward(probe_a.clone()).matmul(encoder.forward(probe_b.clone()).transpose())/0.1,labels.clone()).into_scalar::<f32>();
    for _ in 0..100 {
        let a=encoder.forward(x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device));
        let b=encoder.forward(x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device));
        let loss=loss_fn.forward(a.matmul(b.transpose())/0.1,labels.clone());
        let grads=GradientsParams::from_grads(loss.backward(),&encoder); encoder=optimizer.step(0.01,encoder,grads);
    }
    let after=loss_fn.forward(encoder.forward(probe_a).matmul(encoder.forward(probe_b).transpose())/0.1,labels).into_scalar::<f32>();
    println!("Контрастивная ошибка до {before:.4}, после {after:.4}");
    assert!(after<before*0.8, "Обучение должно снизить ошибку на фиксированной паре видов");
    let new=Tensor::<2>::random([40,8],Distribution::Normal(0.0,1.0),&device);
    let a=encoder.forward(new.clone());
    let b=encoder.forward(new+Tensor::<2>::random([40,8],Distribution::Normal(0.0,0.1),&device));
    let accuracy=a.matmul(b.transpose()).argmax(1).squeeze::<1>().equal(Tensor::<1,Int>::arange(0..40,&device)).float().mean().into_scalar::<f32>();
    println!("Найден свой второй вид: {accuracy:.3}"); assert!(accuracy>0.8);
}
''', 'Один Encoder обрабатывает оба зашумлённых вида; нормированные векторы сравниваются матричным произведением. CrossEntropyLoss обучает совпадение диагональных пар среди 128 объектов. Проверка использует 40 новых объектов.', 'Контрастивная ошибка фиксированных обучающих видов уменьшается минимум на 20%; более 80% новых объектов находят свой второй вид среди 40 кандидатов.', exercise='В обоих обучающих видах и проверочном виде замени стандартное отклонение 0.1 на 1.0 и сравни долю найденных пар.')

    add('kan', r'''
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Edges { coefficients:Param<Tensor<2>> }
impl Edges {
    fn forward(&self,x:Tensor<2>,centers:Tensor<3>)->Tensor<2>{
        let n=x.dims()[0];
        let bases=((x.reshape([n,2,1])-centers)/0.25).powf_scalar(2.0).neg().exp();
        (bases*self.coefficients.val().reshape([1,2,12])).sum_dim(2).sum_dim(1).reshape([n,1])
    }
}
fn main(){
    let device=device().autodiff();
    let x=Tensor::<2>::random([400,2],Distribution::Uniform(-1.0,1.0),&device);
    let y=(x.clone().slice([0..400,0..1]).sin()+x.clone().slice([0..400,1..2]).powf_scalar(2.0)).detach();
    let centers=Tensor::<1>::from_floats((0..12).map(|i|-1.0+2.0*i as f32/11.0).collect::<Vec<_>>().as_slice(),&device).reshape([1,1,12]);
    let mut model=Edges{coefficients:Param::from_tensor(Tensor::<2>::zeros([2,12],&device))};
    let tx=x.clone().slice([320..400]); let ty=y.clone().slice([320..400]);
    let before=ty.clone().powf_scalar(2.0).mean().into_scalar::<f32>();
    let mut optimizer=AdamConfig::new().init();
    for _ in 0..300 {
        let loss=(model.forward(x.clone().slice([0..320]),centers.clone())-y.clone().slice([0..320])).powf_scalar(2.0).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.03,model,grads);
    }
    let error=(model.forward(tx,centers)-ty).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE до {before:.6}, после {error:.6}"); assert!(error<0.01 && error<before*0.1);
}
''','Edges обучает 24 коэффициента гауссовых базисов на двух входных связях; centers фиксированы. 320 точек обучают приближение sin(x₁)+x₂², остальные 80 проверяют его.', 'MSE новых точек ниже 0.01 и как минимум в десять раз меньше исходной.', exercise='Замени y на (x.clone().slice([0..400,0..1])*x.clone().slice([0..400,1..2])).detach(). Edges::forward складывает две независимые кривые f(x₁)+g(x₂), поэтому произведение не представимо этим слоем. Только для этого опыта замени исходный assert!(error<0.01 && error<before*0.1) на assert!(before>0.03 && error>0.03 && error.is_finite());. Выведи error/before: исходная нулевая модель уже оптимальна в среднем для независимых Uniform(-1,1), а её ожидаемая MSE равна 1/9. На конечных выборках ошибки различаются; уменьшение ошибки в десять раз здесь не ожидается.')

    add('linearattention', r'''
fn main(){
    let device=device();
    // Для положительных Uniform(0,1) ELU(x)+1 равно x+1.
    let q=Tensor::<2>::random([12,8],Distribution::Uniform(0.0,1.0),&device)+1.0;
    let k=Tensor::<2>::random([12,8],Distribution::Uniform(0.0,1.0),&device)+1.0;
    let v=Tensor::<2>::random([12,8],Distribution::Uniform(0.0,1.0),&device);
    let mut memory=Tensor::<2>::zeros([8,8],&device); let mut normalizer=Tensor::<2>::zeros([1,8],&device);
    let mut recurrent=Vec::new(); let mut reference=Vec::new();
    for t in 0..12 {
        let qt=q.clone().slice([t..t+1]); let kt=k.clone().slice([t..t+1]);
        memory=memory+kt.clone().transpose().matmul(v.clone().slice([t..t+1])); normalizer=normalizer+kt;
        recurrent.push(qt.clone().matmul(memory.clone())/(qt.clone()*normalizer.clone()).sum_dim(1).add_scalar(1e-6));
        // Только префикс: тот же результат, что нижнетреугольная маска scores.
        let scores=qt.matmul(k.clone().slice([0..t+1]).transpose());
        reference.push(scores.clone().matmul(v.clone().slice([0..t+1]))/(scores.sum_dim(1)+1e-6));
    }
    let error=(Tensor::cat(recurrent,0)-Tensor::cat(reference,0)).abs().max().into_scalar::<f32>();
    println!("Максимальная разница: {error:.8}; память: 64+8 чисел"); assert!(error<1e-5);
}
''','memory накапливает внешние произведения ключей и значений; normalizer — сумму ключей. Прямой расчёт reference использует все разрешённые позиции префикса и сверяется с рекуррентным.', 'Максимальная разница меньше 1e-5; размер накопленной памяти не зависит от длины последовательности.', exercise='Замени длину 12 на 1000 во всех размерах и цикле; сравни размер memory и число оценок scores на последнем шаге.')

    add('moe', r'''
use burn::{nn::LinearConfig,tensor::activation::softmax};
fn main(){
    let device=device(); let x=Tensor::<2>::random([10,4],Distribution::Normal(0.0,1.0),&device);
    let router=LinearConfig::new(4,3).init(&device);
    let experts:Vec<_>=(0..3).map(|_|LinearConfig::new(4,2).init(&device)).collect();
    let scores=softmax(router.forward(x.clone()),1);
    let choice=scores.clone().argmax(1).squeeze::<1>().into_data().try_into_vec_as::<i64>().unwrap();
    let mut counts=[0usize;3]; let mut rows=Vec::new(); let mut error=0.0f32;
    for (row,&expert) in choice.iter().enumerate(){
        let i=expert as usize; counts[i]+=1;
        let answer=experts[i].forward(x.clone().slice([row..row+1]));
        // Проверяем маршрутизацию против независимого вычисления полного батча экспертом.
        error=error.max((answer.clone()-experts[i].forward(x.clone()).slice([row..row+1])).abs().max().into_scalar::<f32>());
        rows.push(answer);
    }
    let answer=Tensor::cat(rows,0);
    let probability_error=(scores.sum_dim(1)-1.0).abs().max().into_scalar::<f32>();
    println!("Выбор {choice:?}; нагрузка {counts:?}; размер {:?}",answer.dims());
    assert_eq!(counts.iter().sum::<usize>(),10); assert_eq!(answer.dims(),[10,2]);
    assert!(error<1e-5 && probability_error<1e-5);
}
''','router выбирает максимум вероятностей трёх экспертов. Для каждой строки вычисляется ответ выбранного Linear; контроль отдельно сверяет его с соответствующей строкой полного вычисления эксперта.', 'Каждый из 10 объектов получает один ответ; сумма нагрузки равна 10, ответы совпадают с выбранными экспертами с ошибкой ниже 1e-5.', exercise='Перед softmax прибавь 10 к первому столбцу оценок router и сравни counts: как меняется нагрузка?')

    add('liquid', r'''
use burn::{nn::LinearConfig,tensor::activation::softplus};
fn main(){
    let device=device(); let target_layer=LinearConfig::new(2,4).init(&device); let speed_layer=LinearConfig::new(2,4).init(&device);
    let x=Tensor::<2>::ones([1,2],&device); let state=Tensor::<2>::zeros([1,4],&device);
    let target=target_layer.forward(x.clone()).tanh(); let speed=softplus(speed_layer.forward(x),1.0);
    let mut distances=Vec::new();
    for dt in [0.0,0.01,0.1,1.0] {
        let decay=(speed.clone()*(-dt)).exp();
        let next=decay.clone()*state.clone()+(decay.neg()+1.0)*target.clone();
        let distance=(next.clone()-target.clone()).powf_scalar(2.0).sum().into_scalar::<f32>(); distances.push(distance);
        if dt==0.0 { assert!((next.clone()-state.clone()).abs().max().into_scalar::<f32>()<1e-7); }
        println!("Интервал {dt}, состояние {:?}, расстояние² {distance:.6}",next.into_data());
    }
    assert!(distances[0]>1e-7); assert!(distances.windows(2).all(|d|d[1]<d[0]));
}
''','target_layer задаёт целевое состояние, speed_layer через softplus — положительную скорость. Экспонента переводит dt в сохранённую долю памяти; каждый опыт начинает с одного state.', 'При dt=0 состояние сохраняется; при 0.01, 0.1 и 1.0 расстояние до target строго уменьшается.', exercise='Инициализируй state другим постоянным значением, например Tensor::<2>::ones([1,4],&device), и измени вход x. Проверь равенство состояния при dt=0 и убывание расстояния при положительных интервалах.')
