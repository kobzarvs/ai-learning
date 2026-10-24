use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor, TensorData}};
fn main() {
    // Flex считает на CPU. autodiff включает вычисление градиентов.
    let device = Device::flex().autodiff();
    let x = Tensor::<2>::from_data(TensorData::new(vec![-1.0_f32, 0., 1., 2.], [4, 1]), &device);
    let y = x.clone() * 3.0 + 2.0;
    let mut model = LinearConfig::new(1, 1).init(&device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        let loss = (model.forward(x.clone()) - y.clone()).powf_scalar(2.0).mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.05, model, gradients);
    }
    let test = Tensor::<2>::from_data([[3.0_f32]], &device);
    let prediction = model.forward(test).into_scalar::<f32>();
    println!("Для x=3 получено {prediction:.4}; ожидаем 11");
    assert!((prediction - 11.0).abs() < 0.1);
}
