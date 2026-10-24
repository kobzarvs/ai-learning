use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams}};
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([300, 2], Distribution::Normal(0.0, 1.0), &device);
    let y = x.clone().slice([0..300, 0..1])*3.0
        - x.clone().slice([0..300, 1..2])*2.0 + 0.7
        + Tensor::<2>::random([300, 1], Distribution::Normal(0.0, 0.1), &device);
    let mut model = LinearConfig::new(2, 1).init(&device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        let loss = (model.forward(x.clone().slice([0..240]))-y.clone().slice([0..240]))
            .powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let error = (model.forward(x.slice([240..300]))-y.slice([240..300]))
        .powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка на 60 новых строках: {error}");
    println!("Коэффициенты: {:?}; сдвиг: {:?}",
        model.weight.val().into_data().try_to_vec::<f32>().unwrap(),
        model.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap());
    assert!(error < 0.06, "Модель не выучила зависимость");
}
