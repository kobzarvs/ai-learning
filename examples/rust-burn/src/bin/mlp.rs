use burn::{module::Module, nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{Device, Tensor, TensorData, activation::relu}};
#[derive(Module, Debug)]
struct Model { hidden: Linear, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.output.forward(relu(self.hidden.forward(x)))
    }
}
fn main() {
    let device = Device::flex().autodiff();
    let values: Vec<f32> = (0..80).map(|i| i as f32/40.0-1.0).collect();
    let x = Tensor::<2>::from_data(TensorData::new(values, [80, 1]), &device);
    let y = x.clone().powf_scalar(2.0);
    let mut model = Model { hidden: LinearConfig::new(1, 24).init(&device),
        output: LinearConfig::new(24, 1).init(&device) };
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..500 {
        let loss = (model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, gradients);
    }
    let test = Tensor::<2>::from_data([[-0.735_f32], [0.315], [0.825]], &device);
    let error = (model.forward(test.clone())-test.powf_scalar(2.0)).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка на новых числах: {error:.6}");
    assert!(error < 0.02);
}
