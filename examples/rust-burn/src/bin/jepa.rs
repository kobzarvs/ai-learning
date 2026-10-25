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
    let x = Tensor::<2>::random([256,4],Distribution::Normal(0.0,1.0),&device);

    // Берём часть тензора по указанным диапазонам индексов.
    let context = x.slice([0..256,0..2]);
    let a = context.clone().slice([0..256,0..1]);
    let b = context.clone().slice([0..256,1..2]);
    let target_part = Tensor::cat(vec![a.clone()+b.clone(),a-b],1);
    let target_encoder = LinearConfig::new(2,8).with_bias(false).init(&device).no_grad();

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let target = target_encoder.forward(target_part).detach();
    let target_variance = (target.clone()-target.clone().mean_dim(0)).powf_scalar(2.0).mean().into_scalar::<f32>();
    let mut predictor = Model::new(2,24,8,&device);
    let before = mse(predictor.forward(context.clone().slice([200..256])),target.clone().slice([200..256]));
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.0).init();
    for _ in 0..200 {
        let loss = (predictor.forward(context.clone().slice([0..200]))-target.clone().slice([0..200])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &predictor);
        predictor = opt.step(0.02,predictor,grads);
    }
    let after = mse(predictor.forward(context.slice([200..256])),target.slice([200..256]));
    println!("Ошибка до: {before}; после: {after}; дисперсия цели: {target_variance}");
    assert!(target_variance>0.05 && after<before*0.1 && after<0.03);
}
