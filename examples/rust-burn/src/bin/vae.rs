use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamWConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Vae { encoder: Linear, decoder: Linear }
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let x = Tensor::<2>::random([256,2], Distribution::Normal(0.0,0.3), &device)
        + Tensor::<2>::from_data([[1.0_f32,-1.0]], &device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = Vae { encoder: LinearConfig::new(2,4).init(&device), decoder: LinearConfig::new(2,2).init(&device) };
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01).init();
    for _ in 0..200 {
        let encoded = model.encoder.forward(x.clone());

        // Берём часть тензора по указанным диапазонам индексов.
        let mu = encoded.clone().slice([0..256,0..2]);
        let logvar = encoded.slice([0..256,2..4]).clamp(-8.0,8.0);
        let z = mu.clone()+(logvar.clone()*0.5).exp()*Tensor::<2>::random([256,2],Distribution::Normal(0.0,1.0),&device);
        let reconstruction = (model.decoder.forward(z)-x.clone()).powf_scalar(2.0).mean();
        let kl = (mu.powf_scalar(2.0)+logvar.clone().exp()-1.0-logvar).mean()*0.5;
        let loss = reconstruction+kl*0.1;

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.03,model,grads);
    }
    let encoded = model.encoder.forward(x.clone());
    let z = encoded.clone().slice([0..256,0..2])+(encoded.slice([0..256,2..4]).clamp(-8.0,8.0)*0.5).exp()
        *Tensor::<2>::random([256,2],Distribution::Normal(0.0,1.0),&device);

    // Читаем единственное значение тензора как обычное число Rust.
    let error = (model.decoder.forward(z)-x).powf_scalar(2.0).mean().into_scalar::<f32>();
    let samples = model.decoder.forward(Tensor::<2>::random([1000,2],Distribution::Normal(0.0,1.0),&device));
    let mean = samples.clone().mean_dim(0).into_data().try_to_vec::<f32>().unwrap();
    println!("Новые точки: {:?}; ошибка: {error}; среднее: {mean:?}", samples.slice([0..5]).into_data());
    assert!(error < 0.15 && (mean[0]-1.0).abs()<0.3 && (mean[1]+1.0).abs()<0.3);
}
