"""Scientific mechanisms, each preserving the Python lesson's computation."""


def register(add):
    add('pinn lab-science', r'''
use burn::{module::{Module,Param}, optim::{AdamWConfig,GradientsParams}, tensor::activation::tanh};
#[derive(Module,Debug)]
struct Field { w: Param<Tensor<2>>, b: Param<Tensor<2>>, v: Param<Tensor<2>>, c: Param<Tensor<2>> }
impl Field {
    fn value_derivative(&self,x: Tensor<2>) -> (Tensor<2>,Tensor<2>) {
        let h = tanh(x.matmul(self.w.val())+self.b.val());
        let u = h.clone().matmul(self.v.val())+self.c.val();
        // Точная цепочка производных tanh: все операции остаются в графе Burn.
        let du = (h.powf_scalar(2.0).neg()+1.0)*self.w.val();
        (u,du.matmul(self.v.val()))
    }
}
fn main() {
    let device = device().autodiff();
    let mut model = Field {
        w: Param::from_tensor(Tensor::random([1,24],Distribution::Normal(0.0,0.4),&device)),
        b: Param::from_tensor(Tensor::zeros([1,24],&device)),
        v: Param::from_tensor(Tensor::random([24,1],Distribution::Normal(0.0,0.2),&device)),
        c: Param::from_tensor(Tensor::zeros([1,1],&device)),
    };
    let mut opt = AdamWConfig::new().with_weight_decay(0.01).with_epsilon(1e-8).init();
    for _ in 0..600 {
        let x = Tensor::<2>::random([80,1],Distribution::Uniform(0.0,2.0),&device);
        let (u,du) = model.value_derivative(x);
        let equation_error = (du+u).powf_scalar(2.0).mean();
        let boundary_error = (model.value_derivative(Tensor::zeros([1,1],&device)).0-1.0).powf_scalar(2.0).mean();
        let loss = equation_error+boundary_error;
        let grads = GradientsParams::from_grads(loss.backward(),&model);
        model = opt.step(0.01,model,grads);
    }
    let x = Tensor::<1>::from_data((0..50).map(|i|2.0*i as f32/49.0).collect::<Vec<_>>().as_slice(),&device).reshape([50,1]);
    let (u,du) = model.value_derivative(x.clone());
    let residual = (du+u.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let error = (u-(-x).exp()).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка exp(-x): {error}; невязка: {residual}");
    assert!(error<0.002 && residual<0.005);
}
''', 'Field обучает 24 tanh-нейрона по уравнению u′+u=0 и условию u(0)=1. value_derivative вычисляет точную производную по правилу цепочки средствами Burn; backward дифференцирует эту формулу по параметрам. Это эквивалент производной с create_graph в Python без необходимости вторичного backward. exp используется только в проверке.',
        'На 50 проверочных координатах MSE меньше 0.002, невязка меньше 0.005. Опыт ограничен одним простым ОДУ.',
        exercise='Замени let loss = equation_error+boundary_error; на let loss = equation_error; и удали неиспользуемое объявление boundary_error. Только для этого опыта замени итоговый assert!(error<0.002 && residual<0.005) на assert!(error.is_finite() && residual<0.005);. Выведи также model.value_derivative(Tensor::zeros([1,1],&device)).0.into_scalar::<f32>() как u(0). Без граничного условия любая кривая C*exp(-x), включая u=0, решает уравнение, поэтому близость к exp(-x) больше не обязательна. Сравни error, residual и u(0) с исходным запуском.')

    add('neuraloperator', r'''
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Spectrum { real: Param<Tensor<2>>, imag: Param<Tensor<2>> }
impl Spectrum {
    fn forward(&self,x: Tensor<2>,cos: Tensor<2>,sin: Tensor<2>,inverse_cos: Tensor<2>,inverse_sin: Tensor<2>) -> Tensor<2> {
        let re = x.clone().matmul(cos); let im = x.matmul(sin).neg();
        let out_re = re.clone()*self.real.val()-im.clone()*self.imag.val();
        let out_im = re*self.imag.val()+im*self.real.val();
        out_re.matmul(inverse_cos)-out_im.matmul(inverse_sin)
    }
}
fn main() {
    let device = device().autodiff();
    // Явный DFT той же длины 32: заменяет FFT, сохраняя комплексные коэффициенты.
    let mut c = Vec::new(); let mut s = Vec::new();
    for n in 0..32 { for k in 0..17 { let a = std::f32::consts::TAU*n as f32*k as f32/32.0; c.push(a.cos()); s.push(a.sin()); } }
    let cos = Tensor::<1>::from_data(c.as_slice(),&device).reshape([32,17]);
    let sin = Tensor::<1>::from_data(s.as_slice(),&device).reshape([32,17]);
    let factors = Tensor::<1>::from_data((0..17).map(|k|if k==0 || k==16 {1.0_f32/32.0} else {2.0/32.0}).collect::<Vec<_>>().as_slice(),&device).reshape([17,1]);
    let ic = cos.clone().transpose()*factors.clone(); let is = sin.clone().transpose()*factors;
    let x = Tensor::<2>::random([180,32],Distribution::Normal(0.0,1.0),&device);
    let response = Tensor::<1>::from_data((0..17).map(|k|(-(k as f32)/4.0).exp()).collect::<Vec<_>>().as_slice(),&device).reshape([1,17]);
    let y = ((x.clone().matmul(cos.clone())*response.clone()).matmul(ic.clone())+(x.clone().matmul(sin.clone())*response).matmul(is.clone())).detach();
    let mut model = Spectrum {real:Param::from_tensor(Tensor::ones([1,17],&device)),imag:Param::from_tensor(Tensor::zeros([1,17],&device))};
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    let eval = |m: &Spectrum| (m.forward(x.clone().slice([140..180]),cos.clone(),sin.clone(),ic.clone(),is.clone())-y.clone().slice([140..180])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = eval(&model);
    for _ in 0..180 {
        let loss = (model.forward(x.clone().slice([0..140]),cos.clone(),sin.clone(),ic.clone(),is.clone())-y.clone().slice([0..140])).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.05,model,grads);
    }
    let after = eval(&model); println!("Ошибка новых сигналов: {before} -> {after}");
    assert!(after<0.0001 && after<before*0.01);
}
''', 'Явные матрицы DFT и обратного DFT заменяют rfft/irfft для 32 чисел. Spectrum обучает действительную и мнимую части 17 комплексных весов Adam, используя только 140 сигналов. Оставшиеся 40 проверяют сглаживание. Это линейный спектральный оператор, не полный FNO.',
        'Проверочная MSE меньше 0.0001 и уменьшается минимум в 100 раз.',
        exercise='Измени знаменатель 4.0 в response на 8.0. Сравни обученные model.real и ошибку after.')

    add('symbolic hybrids', r'''
use burn::tensor::activation::softmax;
fn main() {
    let device = device();
    let logits = Tensor::<2>::from_data([[1.0_f32,2.0,5.0]],&device);
    let forbidden = Tensor::<2,burn::tensor::Bool>::from_data([[false,false,true]],&device);
    let restricted = logits.mask_fill(forbidden,f32::NEG_INFINITY);
    let probabilities = softmax(restricted,1);
    let action = probabilities.clone().argmax(1).into_scalar::<i32>();
    let p = probabilities.into_data().try_to_vec::<f32>().unwrap();
    println!("Вероятности: {p:?}; выбрано: {}",["читать","отправить","удалить"][action as usize]);
    assert_eq!(action,1); assert_eq!(p[2],0.0); assert!((p.iter().sum::<f32>()-1.0).abs()<1e-6);
}
''', 'Маска запрещает удаление до softmax. Максимальная исходная оценка 5 не может обойти правило; обучения и логического вывода здесь нет.',
        'Вероятности примерно [0.269, 0.731, 0]; выбирается отправка.',
        exercise='Сделай forbidden полностью true. Перед mask_fill проверь, что есть разрешённое действие, и выведи отказ вместо softmax.')

    add('symbolicreg', r'''
fn feature(x:f64,kind:usize)->f64 { match kind {0=>x,1=>x*x,_=>x.sin()} }
fn main() {
    let mut errors = Vec::new();
    for kind in 0..3 {
        let (mut sf,mut sy,mut sff,mut sfy) = (0.0,0.0,0.0,0.0);
        for i in 0..70 { let x = -2.0+4.0*i as f64/99.0; let f=feature(x,kind); let y=2.0*x*x+0.5; sf+=f;sy+=y;sff+=f*f;sfy+=f*y; }
        // Решение двухпараметрической задачи наименьших квадратов.
        let a = (70.0*sfy-sf*sy)/(70.0*sff-sf*sf); let b = (sy-a*sf)/70.0;
        let error = (70..100).map(|i| {let x=-2.0+4.0*i as f64/99.0; (a*feature(x,kind)+b-(2.0*x*x+0.5)).powi(2)}).sum::<f64>()/30.0;
        println!("{}: a={a}, b={b}, ошибка={error}",["a*x+b","a*x²+b","a*sin(x)+b"][kind]);
        errors.push(error);
        if kind==1 {assert!((a-2.0).abs()<1e-10 && (b-0.5).abs()<1e-10);}
    }
    assert!(errors[1]<1e-20 && errors[0]>1.0 && errors[2]>1.0);
}
''', 'Три структуры формул сравниваются на 30 новых точках; коэффициенты вычисляются по первым 70 методом наименьших квадратов. Для двух столбцов достаточно явного решения системы 2×2 на f64, тензоры Burn здесь не нужны.',
        'Квадратичный кандидат восстанавливает 2 и 0.5; его MSE практически нулевая. Это ограниченный перебор трёх структур.',
        exercise='Добавь в feature отдельную ветку 3=>x*x*x перед веткой по умолчанию, расширь цикл до 0..4 и добавь четвёртую строку a*x³+b в массив имён внутри println!. Сравни проверочную error, а не учебную ошибку.')

    add('evolution', r'''
fn score(x:[f32;2])->f32 {(x[0]-1.5).powi(2)+(x[1]+0.7).powi(2)}
fn main() {
    let device = device();
    let raw = Tensor::<2>::random([100,2],Distribution::Normal(0.0,3.0),&device).into_data().try_to_vec::<f32>().unwrap();
    let mut population:Vec<[f32;2]> = raw.chunks_exact(2).map(|p|[p[0],p[1]]).collect();
    let initial = population.iter().map(|p|score(*p)).fold(f32::INFINITY,f32::min);
    let mut previous = initial;
    for _ in 0..50 {
        population.sort_by(|a,b|score(*a).total_cmp(&score(*b)));
        let elite = population[..10].to_vec();
        let noise = Tensor::<2>::random([90,2],Distribution::Normal(0.0,0.08),&device).into_data().try_to_vec::<f32>().unwrap();
        population = elite.clone();
        for i in 0..90 {population.push([elite[i%10][0]+noise[2*i],elite[i%10][1]+noise[2*i+1]]);}
        let current = population.iter().map(|p|score(*p)).fold(f32::INFINITY,f32::min);
        assert!(current<=previous); previous=current;
    }
    population.sort_by(|a,b|score(*a).total_cmp(&score(*b)));
    println!("Лучший: {:?}; ошибка {initial} -> {}",population[0],score(population[0]));
    assert!(score(population[0])<0.001);
}
''', 'Burn создаёт начальную нормальную популяцию и мутации. Rust сортирует оценки, сохраняет 10 элитных особей и создаёт 90 потомков в каждом из 50 поколений. Производные не вычисляются.',
        'Квадрат расстояния до [1.5, −0.7] меньше 0.001; элитизм не допускает ухудшения лучшего результата.',
        exercise='Замени постоянное 0.08 в Distribution::Normal на убывающее по номеру поколения значение от 0.5 до 0.01. Сравни previous.')

    add('snn', r'''
fn simulate(current:f32)->Vec<i32> {
    let device = device(); let mut voltage = Tensor::<1>::zeros([1],&device); let mut spikes=Vec::new();
    for _ in 0..20 {
        voltage=voltage*0.9+current;
        let spike=voltage.clone().greater_equal_elem(1.0).float();
        voltage=voltage-spike.clone(); spikes.push(spike.into_scalar::<f32>() as i32);
        let v=voltage.clone().into_scalar::<f32>(); assert!(v>=0.0 && v<1.0);
    }
    spikes
}
fn main() {
    let spikes=simulate(0.4); let count:i32=spikes.iter().sum();
    println!("Импульсы: {spikes:?}; всего: {count}");
    assert_eq!(spikes,vec![0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]);
    assert!(simulate(0.2).iter().sum::<i32>()<count && simulate(0.8).iter().sum::<i32>()>count);
}
''', 'Тензор voltage накапливает ток с утечкой 0.9; пороговый тензор spike вычитается после импульса. Это прямой расчёт LIF без обучения.',
        'При токе 0.4 возникает шесть импульсов за 20 шагов. Больший ток увеличивает частоту.',
        exercise='Измени коэффициент 0.9 в simulate на 0.7 и сравни число импульсов для токов 0.2, 0.4 и 0.8. Обнови ожидаемую последовательность в assert_eq!.')

    add('quantum', r'''
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Qubit {angle:Param<Tensor<1>>}
fn main() {
    let device = device().autodiff();
    let mut qubit=Qubit {angle:Param::from_tensor(Tensor::full([1],0.2,&device))};
    let mut opt=AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..150 {
        let probability_one=(qubit.angle.val()/2.0).sin().powf_scalar(2.0);
        let loss=(probability_one-0.8).powf_scalar(2.0).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&qubit); qubit=opt.step(0.05,qubit,grads);
    }
    let angle=qubit.angle.val().into_scalar::<f32>();
    let p=(qubit.angle.val()/2.0).sin().powf_scalar(2.0).into_scalar::<f32>();
    let p0=(angle/2.0).cos().powi(2);
    println!("Угол: {angle}; вероятность 1: {p}");
    assert!((p-0.8).abs()<0.003 && (p+p0-1.0).abs()<1e-6);
}
''', 'Adam действительно обучает угол RY по вероятности sin²(angle/2). Burn хранит параметр и вычисляет градиент; это классическая симуляция одного кубита, без квантового устройства.',
        'Вероятность единицы отличается от 0.8 менее чем на 0.003; вероятности двух исходов суммируются в 1.',
        exercise='После обучения создай Tensor::<1>::random([100], Distribution::Uniform(0.0,1.0), &device), сравни его с p через lower_elem и усредни float(). Это 100 испытаний Бернулли; сравни частоту с p.')

    add('hdc', r'''
fn main() {
    let device=device();
    let x=Tensor::<2>::random([300,2],Distribution::Normal(0.0,1.0),&device);
    let points=x.clone().into_data().try_to_vec::<f32>().unwrap();
    let labels:Vec<usize>=points.chunks_exact(2).map(|p|usize::from(p[0]+0.5*p[1]>0.0)).collect();
    let projection=Tensor::<2>::random([2,1024],Distribution::Normal(0.0,1.0),&device);
    let codes=x.matmul(projection).greater_equal_elem(0.0).float()*2.0-1.0;
    let raw=codes.clone().into_data().try_to_vec::<f32>().unwrap();
    let mut proto=vec![0.0_f32;2048]; let mut counts=[0;2];
    for i in 0..240 {counts[labels[i]]+=1; for j in 0..1024 {proto[labels[i]*1024+j]+=raw[i*1024+j];}}
    assert!(counts.iter().all(|n|*n>50));
    let prototypes=Tensor::<1>::from_data(proto.as_slice(),&device).reshape([2,1024]);
    let norm=prototypes.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-8);
    let prototypes=prototypes/norm;
    let scores=(codes.slice([240..300])/32.0).matmul(prototypes.transpose());
    let predicted=scores.argmax(1).into_data().try_to_vec::<i32>().unwrap();
    let accuracy=predicted.iter().zip(&labels[240..]).filter(|(a,b)|**a as usize==**b).count() as f32/60.0;
    println!("Точность новых точек: {accuracy}"); assert!(accuracy>0.85);
}
''', 'Знаки случайной проекции Burn дают 1024-битные биполярные коды. Первые 240 размеченных кодов суммируются в два прототипа, нормируются и сравниваются с последними 60 через скалярное произведение. Градиентного обучения нет.',
        'Точность на 60 новых точках выше 0.85. Это простая геометрия; случайные коды сами по себе не создают семантику.',
        exercise='Замени размер 1024 на 32 во всех формах, индексах и циклах; нормировку кодов 32.0 замени на sqrt(32). Затем сравни accuracy с размером 4096.')
