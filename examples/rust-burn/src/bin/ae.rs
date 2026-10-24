use burn::{module::Module, nn::{Linear, LinearConfig}, optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor, TensorData}};
#[derive(Module, Debug)]
struct Autoencoder { encode: Linear, decode: Linear }
impl Autoencoder {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> { self.decode.forward(self.encode.forward(x)) }
}
fn main() {
    let device = Device::flex().autodiff();
    let values: Vec<f32> = (0..60).flat_map(|i| { let t=i as f32/30.0-1.0; [t, 2.0*t, -t] }).collect();
    let x=Tensor::<2>::from_data(TensorData::new(values, [60, 3]), &device);
    let mut model=Autoencoder { encode: LinearConfig::new(3, 1).init(&device), decode: LinearConfig::new(1, 3).init(&device) };
    let mut optimizer=AdamConfig::new().init();
    for _ in 0..350 {
        let loss=(model.forward(x.clone())-x.clone()).powf_scalar(2.0).mean();
        let gradients=GradientsParams::from_grads(loss.backward(), &model);
        model=optimizer.step(0.03, model, gradients);
    }
    let test=Tensor::<2>::from_data([[0.315_f32, 0.63, -0.315]], &device);
    let error=(model.forward(test.clone())-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка восстановления нового объекта: {error}");
    assert!(error < 0.01);
}
