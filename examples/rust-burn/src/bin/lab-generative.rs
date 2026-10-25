use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamWConfig, GradientsParams}, tensor::activation::silu};
#[derive(Module, Debug)]
struct Model { hidden: Linear, output: Linear }
impl Model {
    fn new(input: usize, hidden: usize, output: usize, device: &Device) -> Self {
        // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
        Self { hidden: LinearConfig::new(input, hidden).init(device), output: LinearConfig::new(hidden, output).init(device) }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.output.forward(silu(self.hidden.forward(x))) }
}

fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let mut velocity = Model::new(3,32,2,&device);
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..350 {
        let source = Tensor::<2>::random([128,2],Distribution::Normal(0.0,1.0),&device);
        let target = Tensor::<2>::random([128,2],Distribution::Normal(0.0,0.3),&device)+Tensor::<2>::from_data([[2.0_f32,-1.0]],&device);
        let time = Tensor::<2>::random([128,1],Distribution::Uniform(0.0,1.0),&device);
        let position = (time.clone().neg()+1.0)*source.clone()+time.clone()*target.clone();
        let loss = (velocity.forward(Tensor::cat(vec![position,time],1))-(target-source)).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &velocity);
        velocity = opt.step(0.01,velocity,grads);
    }
    let mut samples = Tensor::<2>::random([500,2],Distribution::Normal(0.0,1.0),&device);
    let steps = 60;
    for step in 0..steps {
        let time = Tensor::<2>::full([500,1],step as f32/steps as f32,&device);
        let update = velocity.forward(Tensor::cat(vec![samples.clone(),time],1))/steps as f32;

        // Отделяем тензор от графа градиентов, сохраняя его значения.
        samples = (samples+update).detach();
    }
    let mean_tensor = samples.clone().mean_dim(0);
    let mean = mean_tensor.clone().into_data().try_to_vec::<f32>().unwrap();
    let std = (samples-mean_tensor).powf_scalar(2.0).mean_dim(0).sqrt().into_data().try_to_vec::<f32>().unwrap();
    println!("Среднее: {mean:?}; разброс: {std:?}");
    assert!((mean[0]-2.0).abs()<0.3 && (mean[1]+1.0).abs()<0.3);
    assert!(std.iter().all(|s| *s>0.12 && *s<0.6));
}
