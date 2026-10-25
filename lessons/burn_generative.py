"""Standalone Burn programs matching the generative Python lessons."""

MLP = r'''
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamWConfig, GradientsParams}, tensor::activation::silu};
#[derive(Module, Debug)]
struct Model { hidden: Linear, output: Linear }
impl Model {
    fn new(input: usize, hidden: usize, output: usize, device: &Device) -> Self {
        Self { hidden: LinearConfig::new(input, hidden).init(device), output: LinearConfig::new(hidden, output).init(device) }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.output.forward(silu(self.hidden.forward(x))) }
}
fn mse(x: Tensor<2>, y: Tensor<2>) -> f32 { (x-y).powf_scalar(2.0).mean().into_scalar() }
'''


def register(add):
    add('anomaly', r'''
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Autoencoder { encoder: Linear, decoder: Linear }
impl Autoencoder {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.decoder.forward(self.encoder.forward(x)) }
}
fn main() {
    let device = device().autodiff();
    let normal = Tensor::<2>::random([300,1], Distribution::Normal(0.0,1.0), &device)
        .matmul(Tensor::<2>::from_data([[1.0_f32,2.0,-1.0,0.5]], &device))
        + Tensor::<2>::random([300,4], Distribution::Normal(0.0,0.03), &device);
    let mut model = Autoencoder { encoder: LinearConfig::new(4,1).init(&device), decoder: LinearConfig::new(1,4).init(&device) };
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..300 {
        let train = normal.clone().slice([0..240]);
        let loss = (model.forward(train.clone())-train).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.02, model, grads);
    }
    let test = normal.slice([240..300]);
    let ordinary = (model.forward(test.clone())-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    let strange = Tensor::<2>::from_data([[5.0_f32,-5.0,5.0,-5.0]], &device);
    let unusual = (model.forward(strange.clone())-strange).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Обычные: {ordinary}; необычный: {unusual}");
    assert!(ordinary < 0.02 && unusual > 10.0 && unusual > ordinary*100.0);
}
''', 'Два линейных слоя сжимают четыре связанных признака до одного числа и восстанавливают их. Adam обучается только на первых 240 строках; последние 60 строк и нарушающий связь объект проверяются отдельно.',
        'Ошибка обычных объектов меньше 0.02; ошибка необычного объекта больше 10 и существенно выше обычной.',
        exercise='Перенеси объявление strange перед циклом обучения. Внутри цикла после объявления train добавь let train = Tensor::cat(vec![train, strange.clone()], 0); чтобы обучаться также на необычном объекте. Сравни ordinary и unusual: обучение на аномалии ослабляет её обнаружение.')

    add('vae', r'''
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamWConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Vae { encoder: Linear, decoder: Linear }
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([256,2], Distribution::Normal(0.0,0.3), &device)
        + Tensor::<2>::from_data([[1.0_f32,-1.0]], &device);
    let mut model = Vae { encoder: LinearConfig::new(2,4).init(&device), decoder: LinearConfig::new(2,2).init(&device) };
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..200 {
        let encoded = model.encoder.forward(x.clone());
        let mu = encoded.clone().slice([0..256,0..2]);
        let logvar = encoded.slice([0..256,2..4]).clamp(-8.0,8.0);
        let z = mu.clone()+(logvar.clone()*0.5).exp()*Tensor::<2>::random([256,2],Distribution::Normal(0.0,1.0),&device);
        let reconstruction = (model.decoder.forward(z)-x.clone()).powf_scalar(2.0).mean();
        let kl = (mu.powf_scalar(2.0)+logvar.clone().exp()-1.0-logvar).mean()*0.5;
        let loss = reconstruction+kl*0.1;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.03,model,grads);
    }
    let encoded = model.encoder.forward(x.clone());
    let z = encoded.clone().slice([0..256,0..2])+(encoded.slice([0..256,2..4]).clamp(-8.0,8.0)*0.5).exp()
        *Tensor::<2>::random([256,2],Distribution::Normal(0.0,1.0),&device);
    let error = (model.decoder.forward(z)-x).powf_scalar(2.0).mean().into_scalar::<f32>();
    let samples = model.decoder.forward(Tensor::<2>::random([1000,2],Distribution::Normal(0.0,1.0),&device));
    let mean = samples.clone().mean_dim(0).into_data().try_to_vec::<f32>().unwrap();
    println!("Новые точки: {:?}; ошибка: {error}; среднее: {mean:?}", samples.slice([0..5]).into_data());
    assert!(error < 0.15 && (mean[0]-1.0).abs()<0.3 && (mean[1]+1.0).abs()<0.3);
}
''', 'encoder выдаёт mu и logvar; выборка z через репараметризацию сохраняет градиент. decoder обучается вместе с encoder по сумме MSE и 0.1 KL. Новые точки получаем из стандартного нормального шума.',
        'Ошибка восстановления меньше 0.15, среднее новых точек близко к [1, −1]. Это линейный VAE, не генератор изображений.',
        exercise='В выражении reconstruction+kl*0.1 замени 0.1 на 0.001 и затем на 1.0. Сравни error и разброс samples.')

    add('gan', MLP.replace('fn mse(x: Tensor<2>, y: Tensor<2>) -> f32 { (x-y).powf_scalar(2.0).mean().into_scalar() }\n', '')+r'''
fn softplus(x: Tensor<2>) -> Tensor<2> {
    x.clone().clamp_min(0.0)+((-x.abs()).exp()+1.0).log()
}
fn real_loss(logits: Tensor<2>) -> Tensor<1> { softplus(-logits).mean() }
fn fake_loss(logits: Tensor<2>) -> Tensor<1> { softplus(logits).mean() }
fn main() {
    let device = device().autodiff();
    let mut generator = Model::new(1,12,1,&device);
    let mut critic = Model::new(1,12,1,&device);
    let mut g_opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    let mut d_opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..300 {
        let real = Tensor::<2>::random([64,1],Distribution::Normal(2.0,0.4),&device);
        let noise = Tensor::<2>::random([64,1],Distribution::Normal(0.0,1.0),&device);
        let fake = generator.forward(noise.clone());
        let loss = real_loss(critic.forward(real))+fake_loss(critic.forward(fake.detach()));
        let grads = GradientsParams::from_grads(loss.backward(), &critic);
        critic = d_opt.step(0.01,critic,grads);
        // Замороженная копия сохраняет градиент по входу, но не по весам критика.
        let frozen = critic.clone().no_grad();
        let loss = real_loss(frozen.forward(generator.forward(noise)));
        let grads = GradientsParams::from_grads(loss.backward(), &generator);
        generator = g_opt.step(0.01,generator,grads);
    }
    let generated = generator.forward(Tensor::<2>::random([1000,1],Distribution::Normal(0.0,1.0),&device));
    let mean = generated.clone().mean().into_scalar::<f32>();
    let std = (generated-mean).powf_scalar(2.0).mean().sqrt().into_scalar::<f32>();
    println!("Среднее: {mean}; разброс: {std}");
    assert!((mean-2.0).abs()<0.8, "Генератор не приблизился к данным");
    assert!(std>0.05 && std<1.2, "Потеря разнообразия или чрезмерный разброс");
}
''', 'generator и critic по очереди минимизируют двоичную логистическую ошибку. detach отделяет шаг критика от генератора; no_grad замораживает копию критика на шаге генератора, сохраняя производную по его входу. Проверяем среднее и разнообразие 1000 новых чисел.',
        'Цель распределения: среднее 2 и разброс 0.4. Короткий GAN может колебаться; проверки допускают отклонение среднего до 0.8, но запрещают почти постоянный ответ.',
        exercise='Увеличь скрытый размер только critic в Model::new с 12 до 64. Сравни mean и std при том же числе шагов.')

    add('diffusion', MLP+r'''
fn main() {
    let device = device().autodiff();
    let mut model = Model::new(2,32,1,&device);
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..250 {
        let clean = Tensor::<2>::random([128,1],Distribution::Normal(2.0,0.3),&device);
        let sigma = Tensor::<2>::random([128,1],Distribution::Uniform(0.05,0.85),&device);
        let noisy = clean.clone()+sigma.clone()*Tensor::<2>::random([128,1],Distribution::Normal(0.0,1.0),&device);
        let loss = (model.forward(Tensor::cat(vec![noisy,sigma],1))-clean).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.01,model,grads);
    }
    let clean = Tensor::<2>::random([100,1],Distribution::Normal(2.0,0.3),&device);
    let noisy = clean.clone()+Tensor::<2>::random([100,1],Distribution::Normal(0.0,0.5),&device);
    let before = mse(noisy.clone(),clean.clone());
    let restored = model.forward(Tensor::cat(vec![noisy,Tensor::<2>::full([100,1],0.5,&device)],1));
    let after = mse(restored,clean);
    println!("До очистки: {before}; после: {after}");
    assert!(after < before*0.65 && after<0.13);
}
''', 'На каждом шаге создаются новые числа и уровни шума. Сеть получает пару noisy, sigma и восстанавливает clean. Проверка использует отдельные 100 чисел с sigma=0.5.',
        'На новых числах ошибка после очистки меньше 65% исходной и меньше 0.13. Это очиститель, полной диффузионной генерации здесь нет.',
        exercise='Удали sigma из обоих Tensor::cat и измени Model::new(2,32,1) на Model::new(1,32,1). Сравни очистку при sigma=0.1 и sigma=0.8.')

    add('autoregression', r'''
use burn::{module::Module, nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig}, optim::{AdamWConfig, GradientsParams}, tensor::Int};
#[derive(Module, Debug)]
struct Language { embedding: Embedding, output: Linear }
impl Language {
    fn forward(&self, x: Tensor<2,Int>) -> Tensor<2> { self.output.forward(self.embedding.forward(x).squeeze_dim::<2>(1)) }
}
fn main() {
    let device = device().autodiff();
    let tokens: Vec<i32> = "abcabcabcabcabcabc".bytes().map(|c|(c-b'a') as i32).collect();
    let x = Tensor::<1,Int>::from_data(tokens[..17].as_ref(),&device).reshape([17,1]);
    let y = Tensor::<1,Int>::from_data(tokens[1..].as_ref(),&device);
    let mut model = Language { embedding: EmbeddingConfig::new(3,8).init(&device), output: LinearConfig::new(8,3).init(&device) };
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.001).init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    for _ in 0..160 {
        let loss = loss_fn.forward(model.forward(x.clone()),y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.02,model,grads);
    }
    let accuracy = model.forward(x).argmax(1).squeeze::<1>().equal(y).float().mean().into_scalar::<f32>();
    let mut current = 0_i32;
    let mut generated = String::from("a");
    for _ in 0..12 {
        current = model.forward(Tensor::<2,Int>::from_data([[current]],&device)).argmax(1).into_scalar::<i32>();
        generated.push((b'a'+current as u8) as char);
    }
    println!("Продолжение: {generated}; точность: {accuracy}");
    assert!(accuracy>0.99);
    assert_eq!(generated,"abcabcabcabca");
}
''', 'Обучаемые Embedding и Linear предсказывают следующий символ по одному предыдущему. После обучения каждый выбранный argmax становится единственным входом следующего шага, без доступа к будущему тексту.',
        'Точность переходов равна 1; самостоятельное продолжение: abcabcabcabca.',
        exercise='Замени строку tokens на abacabab. Перед x создай let n=tokens.len()-1; используй tokens[..n] и reshape([n,1]) вместо 17. У a три перехода в b и один в c: модель одного символа не может правильно ответить более чем на 6 из 7 переходов. Только для этого опыта замени оба исходных assert (accuracy>0.99 и точное равенство продолжения) на assert!(accuracy>=5.0/7.0 && accuracy<=6.0/7.0+1e-6); assert_eq!(generated.len(),13); assert!(generated.bytes().all(|c| b"abc".contains(&c)));. Выведи продолжение и объясни, почему одного предыдущего символа недостаточно.')

    add('flows', r'''
use burn::{module::{Module,Param}, optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Flow { shift: Param<Tensor<1>>, log_scale: Param<Tensor<1>> }
fn main() {
    let device = device().autodiff();
    let data = Tensor::<1>::random([300],Distribution::Normal(3.0,0.5),&device);
    let mut flow = Flow { shift: Param::from_tensor(Tensor::zeros([1],&device)), log_scale: Param::from_tensor(Tensor::zeros([1],&device)) };
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..300 {
        let z = (data.clone()-flow.shift.val())*(-flow.log_scale.val()).exp();
        let loss = (z.powf_scalar(2.0)*0.5+flow.log_scale.val()).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &flow);
        flow = opt.step(0.05,flow,grads);
    }
    let shift = flow.shift.val().into_scalar::<f32>();
    let scale = flow.log_scale.val().exp().into_scalar::<f32>();
    let z = Tensor::<1>::random([5],Distribution::Normal(0.0,1.0),&device);
    let samples = z.clone()*scale+shift;
    let error = ((samples-shift)/scale-z).abs().max().into_scalar::<f32>();
    println!("Сдвиг: {shift}; масштаб: {scale}; ошибка обратимости: {error}");
    assert!((shift-3.0).abs()<0.2 && (scale-0.5).abs()<0.15 && error<1e-5);
}
''', 'Adam обучает shift и log_scale по отрицательному логарифму плотности. Слагаемое log_scale учитывает якобиан растяжения. Обратное преобразование проверяется на свежем шуме.',
        'Сдвиг около 3, масштаб около 0.5; максимальная ошибка обратимости меньше 1e−5.',
        exercise='Удали +flow.log_scale.val() из loss. Только для этого опыта замени исходный составной assert на assert!(shift.is_finite() && scale.is_finite() && scale>0.5 && error<1e-5);: масштаб теперь может расти, поэтому ожидание scale около 0.5 не подходит. Сохрани проверку обратимости и сравни scale с исходным запуском: без поправки на объём увеличение масштаба искусственно снижает ошибку.')

    add('flowmatching lab-generative', MLP.replace('fn mse(x: Tensor<2>, y: Tensor<2>) -> f32 { (x-y).powf_scalar(2.0).mean().into_scalar() }\n', '')+r'''
fn main() {
    let device = device().autodiff();
    let mut velocity = Model::new(3,32,2,&device);
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..350 {
        let source = Tensor::<2>::random([128,2],Distribution::Normal(0.0,1.0),&device);
        let target = Tensor::<2>::random([128,2],Distribution::Normal(0.0,0.3),&device)+Tensor::<2>::from_data([[2.0_f32,-1.0]],&device);
        let time = Tensor::<2>::random([128,1],Distribution::Uniform(0.0,1.0),&device);
        let position = (time.clone().neg()+1.0)*source.clone()+time.clone()*target.clone();
        let loss = (velocity.forward(Tensor::cat(vec![position,time],1))-(target-source)).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &velocity);
        velocity = opt.step(0.01,velocity,grads);
    }
    let mut samples = Tensor::<2>::random([500,2],Distribution::Normal(0.0,1.0),&device);
    let steps = 60;
    for step in 0..steps {
        let time = Tensor::<2>::full([500,1],step as f32/steps as f32,&device);
        let update = velocity.forward(Tensor::cat(vec![samples.clone(),time],1))/steps as f32;
        samples = (samples+update).detach();
    }
    let mean_tensor = samples.clone().mean_dim(0);
    let mean = mean_tensor.clone().into_data().try_to_vec::<f32>().unwrap();
    let std = (samples-mean_tensor).powf_scalar(2.0).mean_dim(0).sqrt().into_data().try_to_vec::<f32>().unwrap();
    println!("Среднее: {mean:?}; разброс: {std:?}");
    assert!((mean[0]-2.0).abs()<0.3 && (mean[1]+1.0).abs()<0.3);
    assert!(std.iter().all(|s| *s>0.12 && *s<0.6));
}
''', 'velocity учится предсказывать target−source в случайной промежуточной позиции и времени. На генерации целевые точки не используются: 60 шагов Эйлера переносят 500 новых шумовых точек. Проверяем и среднее, и разброс.',
        'Среднее около [2, −1] с погрешностью до 0.3; разброс каждого измерения между 0.12 и 0.6 при цели 0.3.',
        exercise='Меняй только steps: 5, 20, 100. Сохрани обученную velocity и одинаковый исходный шум, чтобы сравнить ошибку численного интегрирования.')

    add('ebm', r'''
use burn::{module::{Module,Param}, optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Energy { center: Param<Tensor<1>> }
fn main() {
    let device = device().autodiff();
    let data = Tensor::<1>::random([200],Distribution::Normal(1.5,0.2),&device);
    let mut model = Energy { center: Param::from_tensor(Tensor::zeros([1],&device)) };
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..100 {
        let loss = (data.clone()-model.center.val()).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.05,model,grads);
    }
    let center = model.center.val().detach();
    let learned = center.clone().into_scalar::<f32>();
    let mut x = Tensor::<1>::random([100],Distribution::Normal(0.0,1.0),&device);
    for _ in 0..80 {
        x = x.detach().require_grad();
        let energy = (x.clone()-center.clone()).powf_scalar(2.0)*0.5;
        let grads = energy.sum().backward();
        let gradient = x.grad(&grads).unwrap();
        x = (x-gradient*0.1+Tensor::<1>::random([100],Distribution::Normal(0.0,0.02_f64.sqrt()),&device)).detach();
    }
    let mean = x.clone().mean().into_scalar::<f32>();
    let std = (x-mean).powf_scalar(2.0).mean().sqrt().into_scalar::<f32>();
    println!("Центр: {learned}; среднее образцов: {mean}; разброс: {std}");
    assert!((learned-1.5).abs()<0.12 && (mean-learned).abs()<0.15 && std>0.15 && std<0.5);
}
''', 'Сначала Adam находит центр по 200 значениям. Затем автодифференцирование энергии по x даёт направление каждого из 80 шагов Ланжевена. center отделён от графа; шум поддерживает разнообразие.',
        'Обученный центр около 1.5, среднее образцов рядом с ним. Разброс ненулевой; он задаётся динамикой и шумом, а не обязан равняться учебному 0.2.',
        exercise='Убери слагаемое Tensor::<1>::random из обновления x. Только для этого опыта замени в итоговом assert условие std>0.15 && std<0.5 на std<0.001; обе проверки близости learned к 1.5 и mean к learned сохрани. Сравни std с исходным запуском: 80 детерминированных шагов x←x−0.1(x−center) сжимают разброс в 0.9^80 раза, вместо разнообразных образцов получается одна точка.')

    add('jepa', MLP+r'''
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([256,4],Distribution::Normal(0.0,1.0),&device);
    let context = x.slice([0..256,0..2]);
    let a = context.clone().slice([0..256,0..1]);
    let b = context.clone().slice([0..256,1..2]);
    let target_part = Tensor::cat(vec![a.clone()+b.clone(),a-b],1);
    let target_encoder = LinearConfig::new(2,8).with_bias(false).init(&device).no_grad();
    let target = target_encoder.forward(target_part).detach();
    let target_variance = (target.clone()-target.clone().mean_dim(0)).powf_scalar(2.0).mean().into_scalar::<f32>();
    let mut predictor = Model::new(2,24,8,&device);
    let before = mse(predictor.forward(context.clone().slice([200..256])),target.clone().slice([200..256]));
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.0).init();
    for _ in 0..200 {
        let loss = (predictor.forward(context.clone().slice([0..200]))-target.clone().slice([0..200])).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &predictor);
        predictor = opt.step(0.02,predictor,grads);
    }
    let after = mse(predictor.forward(context.slice([200..256])),target.slice([200..256]));
    println!("Ошибка до: {before}; после: {after}; дисперсия цели: {target_variance}");
    assert!(target_variance>0.05 && after<before*0.1 && after<0.03);
}
''', 'Фиксированный target_encoder превращает связанную скрытую часть в восемь признаков. Только predictor обучается на первых 200 объектах; ошибка проверяется на последних 56. Проверка дисперсии исключает постоянную целевую репрезентацию.',
        'Ошибка на новых объектах уменьшается минимум в десять раз и становится меньше 0.03. Целевые признаки сохраняют ненулевую дисперсию.',
        exercise='Замени target_part на Tensor::<2>::random([256,2],Distribution::Normal(0.0,1.0),&device). Сразу после target создай let constant=target.clone().slice([0..200]).mean_dim(0); let constant_error=(target.clone().slice([200..256])-constant).powf_scalar(2.0).mean().into_scalar::<f32>();. Только для этого опыта замени исходный итоговый assert на assert!(target_variance>0.05 && after.is_finite() && constant_error>0.03); и выведи after рядом с constant_error. Удалены ожидания after<before*0.1 и after<0.03: независимые цели нельзя точно восстановить по context. Не требуй превосходства обученной сети над постоянным прогнозом.')
