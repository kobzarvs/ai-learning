use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Autoencoder { encode: Linear, decode: Linear }
impl Autoencoder {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.decode.forward(self.encode.forward(x)) }
}
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([300, 1], Distribution::Normal(0.0, 1.0), &device)
        .matmul(Tensor::<2>::from_data([[1.0_f32, 2., -1., 0.5]], &device))
        + Tensor::<2>::random([300, 4], Distribution::Normal(0.0, 0.03), &device);
    let mut model = Autoencoder { encode: LinearConfig::new(4, 1).init(&device),
        decode: LinearConfig::new(1, 4).init(&device) };
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        let train = x.clone().slice([0..240]);
        let loss = (model.forward(train.clone())-train).powf_scalar(2.0).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let test = x.slice([240..300]);
    let ordinary = (model.forward(test.clone())-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    let strange = Tensor::<2>::from_data([[5.0_f32, -5., 5., -5.]], &device);
    let unusual = (model.forward(strange.clone())-strange).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Обычные новые объекты: {ordinary}; необычный: {unusual}");
    assert!(ordinary < 0.02 && unusual > ordinary*100.0);
}
