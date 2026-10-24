use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex().autodiff();
    let x = Tensor::<1>::from_data([3.0_f32], &device).require_grad();
    let y = x.clone().powf_scalar(2.0) + x.clone()*2.0;
    let gradients = y.backward();
    let derivative = x.grad(&gradients).unwrap().into_scalar::<f32>();
    println!("Производная: {derivative}");
    assert!((derivative-8.0).abs() < 1e-5);
}
