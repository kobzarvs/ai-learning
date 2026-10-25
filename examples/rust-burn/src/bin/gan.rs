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

fn softplus(x: Tensor<2>) -> Tensor<2> {
    x.clone().clamp_min(0.0)+((-x.abs()).exp()+1.0).log()
}
fn real_loss(logits: Tensor<2>) -> Tensor<1> { softplus(-logits).mean() }
fn fake_loss(logits: Tensor<2>) -> Tensor<1> { softplus(logits).mean() }
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let mut generator = Model::new(1,12,1,&device);
    let mut critic = Model::new(1,12,1,&device);
    let mut g_opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    let mut d_opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..300 {
        let real = Tensor::<2>::random([64,1],Distribution::Normal(2.0,0.4),&device);
        let noise = Tensor::<2>::random([64,1],Distribution::Normal(0.0,1.0),&device);
        let fake = generator.forward(noise.clone());

        // Отделяем тензор от графа градиентов, сохраняя его значения.
        let loss = real_loss(critic.forward(real))+fake_loss(critic.forward(fake.detach()));

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &critic);
        critic = d_opt.step(0.01,critic,grads);

        // Замороженная копия сохраняет градиент по входу, но не по весам критика.
        let frozen = critic.clone().no_grad();
        let loss = real_loss(frozen.forward(generator.forward(noise)));

        // Обратный проход вычисляет градиенты ошибки.
        let grads = GradientsParams::from_grads(loss.backward(), &generator);
        generator = g_opt.step(0.01,generator,grads);
    }
    let generated = generator.forward(Tensor::<2>::random([1000,1],Distribution::Normal(0.0,1.0),&device));

    // Читаем единственное значение тензора как обычное число Rust.
    let mean = generated.clone().mean().into_scalar::<f32>();
    let std = (generated-mean).powf_scalar(2.0).mean().sqrt().into_scalar::<f32>();
    println!("Среднее: {mean}; разброс: {std}");
    assert!((mean-2.0).abs()<0.8, "Генератор не приблизился к данным");
    assert!(std>0.05 && std<1.2, "Потеря разнообразия или чрезмерный разброс");
}
