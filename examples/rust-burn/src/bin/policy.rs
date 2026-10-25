use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams},tensor::activation::softmax};
#[derive(Module,Debug)]
struct Policy { logits: Param<Tensor<1>> }
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();

    // Регистрируем тензор как обучаемый параметр модели.
    let mut model = Policy {logits:Param::from_tensor(Tensor::zeros([3],&device))};

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut opt = AdamConfig::new().init();
    let reward_table = [0.0_f32,0.4,1.0];
    for _ in 0..250 {
        // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
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

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.04,model,grads);
    }
    let p = softmax(model.logits.val(),0).into_data().try_to_vec::<f32>().unwrap();
    println!("Вероятности действий: {p:?}");
    assert!((p.iter().sum::<f32>()-1.0).abs()<1e-5); assert!(p[2]>0.95);
    assert!(p[1]*0.4+p[2]>0.95,"Ожидаемая награда должна улучшиться с 0.4667");
}
