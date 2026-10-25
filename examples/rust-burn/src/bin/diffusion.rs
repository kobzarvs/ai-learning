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

// Читаем единственное значение тензора как обычное число Rust.
fn mse(x: Tensor<2>, y: Tensor<2>) -> f32 { (x-y).powf_scalar(2.0).mean().into_scalar() }

fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let mut model = Model::new(2,32,1,&device);
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..250 {
        let clean = Tensor::<2>::random([128,1],Distribution::Normal(2.0,0.3),&device);
        let sigma = Tensor::<2>::random([128,1],Distribution::Uniform(0.05,0.85),&device);
        let noisy = clean.clone()+sigma.clone()*Tensor::<2>::random([128,1],Distribution::Normal(0.0,1.0),&device);
        let loss = (model.forward(Tensor::cat(vec![noisy,sigma],1))-clean).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
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
