use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module,nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct QNetwork { hidden: Linear, output: Linear }
impl QNetwork { fn forward(&self,x: Tensor<2>) -> Tensor<2> { self.output.forward(silu(self.hidden.forward(x))) } }
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let mut states = vec![]; let mut nexts = vec![]; let mut actions = vec![]; let mut rewards = vec![]; let mut alive = vec![];
    for state in 0..4_i32 { for action in 0..2 {
        let next = (state+if action==1 {1} else {-1}).clamp(0,4);
        for i in 0..5 {states.push(if i==state {1.0_f32} else {0.0}); nexts.push(if i==next {1.0_f32} else {0.0});}
        actions.push(action as i64); rewards.push(if next==4 {1.0_f32} else {-0.02}); alive.push(if next==4 {0.0_f32} else {1.0});
    }}

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let s = Tensor::<1>::from_data(states.as_slice(),&device).reshape([8,5]);
    let ns = Tensor::<1>::from_data(nexts.as_slice(),&device).reshape([8,5]);
    let a = Tensor::<1,Int>::from_data(actions.as_slice(),&device).reshape([8,1]);
    let r = Tensor::<1>::from_data(rewards.as_slice(),&device).reshape([8,1]);
    let mask = Tensor::<1>::from_data(alive.as_slice(),&device).reshape([8,1]);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = QNetwork {hidden:LinearConfig::new(5,16).init(&device),output:LinearConfig::new(16,2).init(&device)};
    let mut opt = AdamWConfig::new().with_weight_decay(0.01).init();
    let mut final_loss = 0.0;
    for _ in 0..300 {
        // Отделяем тензор от графа градиентов, сохраняя его значения.
        let target = (r.clone()+model.forward(ns.clone()).max_dim(1)*mask.clone()*0.95).detach();
        let loss = (model.forward(s.clone()).gather(1,a.clone())-target).powf_scalar(2.0).mean();

        // Читаем единственное значение тензора как обычное число Rust.
        final_loss = loss.clone().into_scalar::<f32>();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.02,model,grads);
    }

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let policy = model.forward(s).argmax(1).reshape([8]).try_into_vec_as::<i64>().unwrap();
    println!("Действия для пар переходов: {policy:?}; Bellman MSE {final_loss}");
    assert!(policy.iter().all(|&a|a==1)); assert!(final_loss<0.01);
    let mut state = 0; let mut steps = 0;
    while state<4 && steps<8 {let a=policy[state*2]; state=if a==1 {state+1} else {state.saturating_sub(1)}; steps+=1;}
    assert_eq!((state,steps),(4,4));
}
