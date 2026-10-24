use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { hidden: Linear, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.output.forward(silu(self.hidden.forward(x)))
    }
}
fn accuracy(model: &Model, x: Tensor<2>, y: Tensor<1, Int>) -> f32 {
    model.forward(x).argmax(1).squeeze::<1>().equal(y).float().mean().into_scalar()
}
fn main() {
    let device = device().autodiff();
    let x = Tensor::<2>::random([320, 2], Distribution::Normal(0.0, 1.0), &device);
    let y = (x.clone().slice([0..320, 0..1])+x.clone().slice([0..320, 1..2])*0.5)
        .greater_elem(0.0).int().squeeze::<1>();
    let mut model = Model { hidden: LinearConfig::new(2, 12).init(&device),
        output: LinearConfig::new(12, 2).init(&device) };
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamConfig::new().init();
    let before = accuracy(&model, x.clone().slice([240..320]), y.clone().slice([240..320]));
    for _ in 0..160 {
        let loss = loss_fn.forward(model.forward(x.clone().slice([0..240])), y.clone().slice([0..240]));
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let after = accuracy(&model, x.slice([240..320]), y.slice([240..320]));
    println!("Доля верных ответов до: {before:.3}; после: {after:.3}");
    assert!(after > 0.9, "Недостаточная точность на новых точках");
}
