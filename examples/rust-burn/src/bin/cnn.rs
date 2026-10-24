use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{TensorData, Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { conv: Conv2d, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let n = x.dims()[0];
        let features = silu(self.conv.forward(x)).mean_dim(2).mean_dim(3).reshape([n, 8]);
        self.output.forward(features)
    }
}
fn main() {
    let device = device().autodiff();
    // 240 картинок, два направления полос, пять возможных сдвигов.
    let labels: Vec<i32> = (0..240).map(|i| i % 2).collect();
    let mut pixels = vec![0.0_f32; 240*64];
    for i in 0..240 {
        let position = 1 + (i/2)%5;
        for row in 0..8 { for col in 0..8 {
            let axis = if labels[i]==0 { col } else { row };
            if axis >= position && axis < position+2 { pixels[i*64+row*8+col]=1.0; }
        }}
    }
    let x = Tensor::<4>::from_data(TensorData::new(pixels, [240, 1, 8, 8]), &device)
        + Tensor::<4>::random([240, 1, 8, 8], Distribution::Normal(0.0, 0.12), &device);
    let y = Tensor::<1, Int>::from_data(TensorData::new(labels, [240]), &device);
    let mut model = Model {
        conv: Conv2dConfig::new([1, 8], [3, 3]).with_padding(PaddingConfig2d::Same).init(&device),
        output: LinearConfig::new(8, 2).init(&device) };
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..160 {
        let loss = loss_fn.forward(model.forward(x.clone().slice([0..180])), y.clone().slice([0..180]));
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let accuracy = model.forward(x.slice([180..240])).argmax(1).squeeze::<1>()
        .equal(y.slice([180..240])).float().mean().into_scalar::<f32>();
    println!("Верно на 60 новых картинках: {accuracy}; обучаемых чисел: {}", model.num_params());
    assert!(accuracy > 0.9, "CNN не научилась различать полосы");
}
