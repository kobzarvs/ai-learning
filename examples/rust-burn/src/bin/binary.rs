use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main(){
    let device=device();
    let weights=Tensor::<2>::random([32,16],Distribution::Normal(0.0,1.0),&device);
    let scale=weights.clone().abs().mean_dim(1);
    let binary=(weights.clone().greater_equal_elem(0.0).float()*2.0-1.0)*scale.clone();
    let x=Tensor::<2>::random([40,16],Distribution::Normal(0.0,1.0),&device);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let error=(x.clone().matmul(binary.clone().transpose())-x.matmul(weights.transpose())).powf_scalar(2.0).mean().into_scalar::<f32>();

    // Читаем единственное значение тензора как обычное число Rust.
    let magnitude_error=(binary.abs()-scale).abs().max().into_scalar::<f32>();
    println!("MSE выхода {error:.5}; теоретические биты знаков 512; фактические байты float32 2048");
    assert_eq!(magnitude_error,0.0); assert!(error>0.1 && error<30.0);
}
