use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let observed = Tensor::<1>::from_data([1.0_f32,1.,0.,1.,1.,0.,1.,1.], &device);
    let success = observed.sum().into_scalar::<f32>();
    let a = 1.0 + success;
    let b = 1.0 + 8.0 - success;
    println!("Средняя вероятность после наблюдений: {}", a/(a+b));
    assert!((a/(a+b)-0.7).abs() < 1e-6);
}
