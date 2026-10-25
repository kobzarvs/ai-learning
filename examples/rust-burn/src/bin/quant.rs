use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::tensor::TensorData;
fn main(){
    let device=device();
    let weights=Tensor::<2>::random([128,64],Distribution::Normal(0.0,0.1),&device);
    let scale=weights.clone().abs().max_dim(1).clamp_min(1e-8)/127.0;
    let rounded=(weights.clone()/scale.clone()).round().clamp(-127.0,127.0);

    // Реальный буфер int8: не считаем float-тензор с целыми значениями сжатым.
    let packed:Vec<i8>=rounded.into_data().try_to_vec::<f32>().unwrap().into_iter().map(|v|v as i8).collect();
    let restored=Tensor::<2>::from_data(TensorData::new(packed.iter().map(|v|*v as f32).collect::<Vec<_>>(),[128,64]),&device)*scale.clone();

    // Читаем единственное значение тензора как обычное число Rust.
    let ratio=((restored.clone()-weights.clone()).abs()/scale).max().into_scalar::<f32>();
    let x=Tensor::<2>::random([20,64],Distribution::Normal(0.0,1.0),&device);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let error=(x.clone().matmul(restored.transpose())-x.matmul(weights.transpose())).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before=128*64*4; let after=packed.len()+128*4;
    println!("MSE выхода {error:.8}; байт до {before}, после с масштабами {after}");
    assert_eq!(after,8704); assert!(after<before/3);
    assert!(ratio<=0.501 && error>0.0 && error<0.001);
}
