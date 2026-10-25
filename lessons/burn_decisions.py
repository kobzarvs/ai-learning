"""Decision learning programs; central catalog provides device and tensor imports."""


def register(add):
    add('controltask imitation', r'''
use burn::{module::Module, nn::{Linear,LinearConfig}, optim::{AdamConfig,GradientsParams}, tensor::activation::tanh};
#[derive(Module,Debug)]
struct Controller { hidden: Linear, output: Linear }
impl Controller {
    fn forward(&self,x: Tensor<2>) -> Tensor<2> { tanh(self.output.forward(tanh(self.hidden.forward(x)))) }
}
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&device);
    let expert = (x.clone().slice([0..400,0..1])*(-0.7)+x.clone().slice([0..400,1..2])*(-0.3)).clamp(-1.0,1.0);
    let mut model = Controller {hidden:LinearConfig::new(2,16).init(&device),output:LinearConfig::new(16,1).init(&device)};
    let error = |m: &Controller| (m.forward(x.clone().slice([320..400]))-expert.clone().slice([320..400])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = error(&model);
    let mut opt = AdamConfig::new().init();
    for _ in 0..400 {
        let loss = (model.forward(x.clone().slice([0..320]))-expert.clone().slice([0..320])).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.02,model,grads);
    }
    let after = error(&model); let mut position = 2.0_f32; let mut speed = 0.0_f32;
    for _ in 0..60 {
        let action = model.forward(Tensor::<2>::from_data([[position,speed]],&device)).into_scalar::<f32>();
        assert!(action.abs()<=1.0); speed += 0.1*action; position += 0.1*speed;
    }
    println!("Ошибка эксперта: {before} -> {after}; положение: {position}");
    assert!(after<0.025 && after<before*0.3);
    assert!(position.abs()<1.8,"Проверка относится к начальному положению 2");
}
''', 'Controller действительно обучает сеть 2→16→1 с tanh копировать ограниченное правило эксперта. Первые 320 состояний используются в Adam, последние 80 — только для MSE. Затем ответы обученной сети управляют скоростью и положением в 60 шагах симуляции.',
        'MSE новых состояний меньше 0.025 и уменьшается как минимум втрое; из положения 2 объект оказывается ближе к нулю. Пересечение цели допустимо.',
        limits='Копирование эксперта в учебном диапазоне не доказывает устойчивость; это компьютерная симуляция, не контроллер реального оборудования.',
        exercise='Замени начальное position = 2.0 на 10.0. Удали только последнюю проверку position.abs()<1.8, привязанную к исходному старту; сохрани проверки MSE и ограничения action. Сравни движение вне учебного диапазона.')

    add('world', r'''
use burn::{nn::LinearConfig,optim::{AdamConfig,GradientsParams}};
fn main() {
    let device = device().autodiff();
    let state = Tensor::<2>::random([400,1],Distribution::Normal(0.0,1.0),&device);
    let action = Tensor::<2>::random([400,1],Distribution::Uniform(-1.0,1.0),&device);
    let next = state.clone()+action.clone()*0.3;
    let x = Tensor::cat(vec![state,action],1);
    let mut model = LinearConfig::new(2,1).init(&device); let mut opt = AdamConfig::new().init();
    let error = |m: &burn::nn::Linear| (m.forward(x.clone().slice([320..400]))-next.clone().slice([320..400])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = error(&model);
    for _ in 0..300 {
        let loss = (model.forward(x.clone().slice([0..320]))-next.clone().slice([0..320])).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.02,model,grads);
    }
    let after = error(&model);
    let candidates = Tensor::<1>::from_data((0..21).map(|i|-1.0+i as f32*0.1).collect::<Vec<_>>().as_slice(),&device).reshape([21,1]);
    let input = Tensor::cat(vec![Tensor::full([21,1],0.8,&device),candidates.clone()],1);
    let index = model.forward(input).powf_scalar(2.0).reshape([21]).argmin(0).into_scalar::<i64>() as usize;
    let chosen = candidates.into_data().try_to_vec::<f32>().unwrap()[index];
    println!("Ошибка прогноза: {before} -> {after}; действие: {chosen}");
    assert!(after<0.001 && after<before*0.01); assert!(chosen<=-0.9);
    assert!((0.8+0.3*chosen).abs()<0.8);
}
''', 'Линейный слой Burn учится на 320 переходах state, action → next. Планировщик сравнивает квадрат прогноза для 21 действия из положения 0.8; реальные коэффициенты динамики используются для данных и проверки, но не заменяют обучение.',
        'На 80 новых переходах MSE меньше 0.001; планировщик выбирает действие около −1, приближающее положение к нулю.',
        exercise='Добавь цикл из пяти шагов планирования: каждый раз подавай последний прогноз как current и заново перебирай candidates. Сохрани одношаговые проверки и отдельно сравни пятишаговый прогноз с истинной динамикой state+0.3*action.')

    add('rl', r'''
use burn::{module::Module,nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct QNetwork { hidden: Linear, output: Linear }
impl QNetwork { fn forward(&self,x: Tensor<2>) -> Tensor<2> { self.output.forward(silu(self.hidden.forward(x))) } }
fn main() {
    let device = device().autodiff();
    let mut states = vec![]; let mut nexts = vec![]; let mut actions = vec![]; let mut rewards = vec![]; let mut alive = vec![];
    for state in 0..4_i32 { for action in 0..2 {
        let next = (state+if action==1 {1} else {-1}).clamp(0,4);
        for i in 0..5 {states.push(if i==state {1.0_f32} else {0.0}); nexts.push(if i==next {1.0_f32} else {0.0});}
        actions.push(action as i64); rewards.push(if next==4 {1.0_f32} else {-0.02}); alive.push(if next==4 {0.0_f32} else {1.0});
    }}
    let s = Tensor::<1>::from_data(states.as_slice(),&device).reshape([8,5]);
    let ns = Tensor::<1>::from_data(nexts.as_slice(),&device).reshape([8,5]);
    let a = Tensor::<1,Int>::from_data(actions.as_slice(),&device).reshape([8,1]);
    let r = Tensor::<1>::from_data(rewards.as_slice(),&device).reshape([8,1]);
    let mask = Tensor::<1>::from_data(alive.as_slice(),&device).reshape([8,1]);
    let mut model = QNetwork {hidden:LinearConfig::new(5,16).init(&device),output:LinearConfig::new(16,2).init(&device)};
    let mut opt = AdamWConfig::new().with_weight_decay(0.01).init();
    let mut final_loss = 0.0;
    for _ in 0..300 {
        let target = (r.clone()+model.forward(ns.clone()).max_dim(1)*mask.clone()*0.95).detach();
        let loss = (model.forward(s.clone()).gather(1,a.clone())-target).powf_scalar(2.0).mean();
        final_loss = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.02,model,grads);
    }
    let policy = model.forward(s).argmax(1).reshape([8]).try_into_vec_as::<i64>().unwrap();
    println!("Действия для пар переходов: {policy:?}; Bellman MSE {final_loss}");
    assert!(policy.iter().all(|&a|a==1)); assert!(final_loss<0.01);
    let mut state = 0; let mut steps = 0;
    while state<4 && steps<8 {let a=policy[state*2]; state=if a==1 {state+1} else {state.saturating_sub(1)}; steps+=1;}
    assert_eq!((state,steps),(4,4));
}
''', 'QNetwork обучается на всех восьми переходах коридора. Цель Bellman использует max следующего Q, скидку 0.95 и нулевую маску терминального перехода; detach запрещает градиент через цель. Gather выбирает только выполненные действия.',
        'Во всех четырёх нетерминальных клетках выбирается вправо; маршрут от 0 достигает 4 за четыре шага, Bellman MSE меньше 0.01.',
        exercise='Перенеси терминал и награду в клетку 0, обучай состояния 1..5, перестрой проверочный rollout со старта 4. Замени ожидаемые действия на 0 и ожидаемый терминал на 0; сохрани проверку четырёх шагов и MSE.')

    add('policy', r'''
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams},tensor::activation::softmax};
#[derive(Module,Debug)]
struct Policy { logits: Param<Tensor<1>> }
fn main() {
    let device = device().autodiff();
    let mut model = Policy {logits:Param::from_tensor(Tensor::zeros([3],&device))};
    let mut opt = AdamConfig::new().init();
    let reward_table = [0.0_f32,0.4,1.0];
    for _ in 0..250 {
        let probability = softmax(model.logits.val(),0);
        let p = probability.clone().into_data().try_to_vec::<f32>().unwrap();
        let uniforms = Tensor::<1>::random([64],Distribution::Uniform(0.0,1.0),&device).into_data().try_to_vec::<f32>().unwrap();
        let actions: Vec<usize> = uniforms.iter().map(|&u|if u<p[0] {0} else if u<p[0]+p[1] {1} else {2}).collect();
        let average = actions.iter().map(|&a|reward_table[a]).sum::<f32>()/64.0;
        // Частоты взвешены преимуществом; действия и награды вне графа.
        let mut weights = [0.0_f32;3];
        for a in actions {weights[a] += (reward_table[a]-average)/64.0;}
        let advantage = Tensor::<1>::from_data(weights,&device);
        let loss = (probability.clamp_min(1e-8).log()*advantage).sum().neg();
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.04,model,grads);
    }
    let p = softmax(model.logits.val(),0).into_data().try_to_vec::<f32>().unwrap();
    println!("Вероятности действий: {p:?}");
    assert!((p.iter().sum::<f32>()-1.0).abs()<1e-5); assert!(p[2]>0.95);
    assert!(p[1]*0.4+p[2]>0.95,"Ожидаемая награда должна улучшиться с 0.4667");
}
''', 'Policy хранит обучаемые logits в Param. Каждый шаг выбирает 64 действия по текущему softmax; веса суммируют (reward−средняя награда) для выбранных действий. Это та же сумма log_prob×advantage, сгруппированная по трём категориям. Градиенты Burn и Adam обновляют logits.',
        'После 250 шагов вероятность третьего действия и ожидаемая награда больше 0.95; вероятности нормированы.',
        exercise='Вместо reward_table[a] сгенерируй независимые случайные награды для 64 действий и храни их вместе с actions. Сравни weights с вычитанием среднего и без него на нескольких seed. Удали проверки p[2]>0.95 и награды>0.95, которые требуют исходной таблицы; сохрани проверку нормировки.')

    add('bandits', r'''
fn main() {
    let device = device();
    // Gamma(k,1) при целом k — сумма k независимых Exp(1).
    // Все posterior shapes целые: исходные единицы плюс Bernoulli-наблюдения.
    let draws = Tensor::<1>::random([1_000_000],Distribution::Uniform(0.0000001,1.0),&device).into_data().try_to_vec::<f32>().unwrap();
    let mut cursor = 0;
    let mut uniform = || {let u=draws[cursor]; cursor+=1; u};
    let chance = [0.2_f32,0.5,0.8]; let mut success = [1_usize;3]; let mut failure = [1_usize;3]; let mut count = [0_usize;3];
    for _ in 0..800 {
        let mut guesses = [0.0_f32;3];
        for a in 0..3 {
            let ga: f32 = (0..success[a]).map(|_|-uniform().ln()).sum();
            let gb: f32 = (0..failure[a]).map(|_|-uniform().ln()).sum();
            guesses[a] = ga/(ga+gb);
        }
        let action = (0..3).max_by(|&a,&b|guesses[a].total_cmp(&guesses[b])).unwrap();
        let reward = uniform()<chance[action];
        if reward {success[action]+=1;} else {failure[action]+=1;} count[action]+=1;
    }
    let means: Vec<f32> = (0..3).map(|a|success[a] as f32/(success[a]+failure[a]) as f32).collect();
    println!("Выборы: {count:?}; posterior means: {means:?}");
    assert_eq!(count.iter().sum::<usize>(),800);
    for a in 0..3 {assert_eq!(success[a]+failure[a],count[a]+2);}
    assert!(count[2]>count[0] && count[2]>count[1] && count[2]>500);
    assert!((means[2]-0.8).abs()<0.08);
}
''', 'Выборка Томпсона начинает с трёх Beta(1,1), берёт независимые posterior-пробы, получает Bernoulli-награду и обновляет успехи/неудачи. Burn генерирует равномерные числа; Gamma целой формы строится суммой −ln(U), Beta — отношением двух Gamma. Это точная конструкция распределения, а не выбор по средним.',
        '800 наблюдений учтены ровно один раз. Третья кнопка выбирается более 500 раз и чаще других; её posterior mean находится в пределах 0.08 от 0.8.',
        limits='Gamma-конструкция здесь рассчитана на целые параметры Beta; сумма экспонент проста, но дороже общего Gamma-сэмплера. Среда стационарная и без контекста.',
        exercise='Сделай chance изменяемым и поменяй chance[0] с chance[2] после шага 400. Отдельно считай выборы в каждой половине. Удали целиком assert!(count[2]>count[0] && count[2]>count[1] && count[2]>500) и проверку means[2] около 0.8: они требуют стационарной среды. Сохрани проверки суммы count и success+failure=count+2; добавь проверку, что сумма выборов в каждой половине равна 400. Выведи долю новой лучшей кнопки 0 в последних 100 шагах и сравни её с первыми 100 после смены, не требуя гарантированного роста в одном случайном запуске.')

    add('selfplay', r'''
fn main() {
    let device = device();
    let payoff = Tensor::<2>::from_data([[0.0_f32,1.0,-1.0],[-1.0,0.0,1.0],[1.0,-1.0,0.0]],&device);
    let mut counts = [9.0_f32,3.0,1.0];
    let exploitability = |c: [f32;3]| {
        let total: f32 = c.iter().sum();
        payoff.clone().matmul(Tensor::<1>::from_data(c.map(|v|v/total),&device).reshape([3,1])).max().into_scalar::<f32>()
    };
    let before = exploitability(counts);
    for _ in 0..2000 {
        let total: f32 = counts.iter().sum();
        let opponent = Tensor::<1>::from_data(counts.map(|v|v/total),&device).reshape([3,1]);
        let values = payoff.clone().matmul(opponent).into_data().try_to_vec::<f32>().unwrap();
        // Как argmax Python: при равенстве выбирается первый индекс.
        let mut action = 0; for a in 1..3 {if values[a]>values[action] {action=a;}}
        counts[action] += 1.0;
    }
    let total: f32 = counts.iter().sum(); let average = counts.map(|v|v/total);
    let after = exploitability(counts);
    println!("Средняя стратегия {average:?}; выгода лучшего ответа {before} -> {after}");
    assert_eq!(total,2013.0); assert!(after<0.05 && after<before);
    assert!(average.iter().all(|v|(v-1.0/3.0).abs()<0.05));
}
''', 'Burn умножает матрицу выигрышей на частоты прошлых ходов. На каждом шаге лучший ответ добавляется в counts; решение не подменяется заранее известным равномерным распределением. Проверяется выгода лучшего ответа против накопленной стратегии.',
        'После 2000 обновлений частоты близки к 1/3, сумма counts равна 2013, выгода лучшего ответа меньше 0.05 и исходной.',
        limits='Это фиктивная игра против собственного усреднённого прошлого в матричной игре: без нейросети, случайных партий и поиска по дереву.',
        exercise='Замени opponent на one-hot последнего выбранного действия, сохраняя counts для итоговой статистики. Выведи первые 12 ответов и проверь трёхходовый цикл. Проверки итоговых частот могут сохраниться: равные частоты цикла сами по себе не означают непредсказуемую игру.')
