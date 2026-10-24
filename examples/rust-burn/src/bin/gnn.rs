use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::nn::LinearConfig;
fn main() {
    let device = device();
    let adjacency=Tensor::<2>::from_data([[1.0_f32,1.,0.,0.],[1.,1.,1.,0.],[0.,1.,1.,0.],[0.,0.,0.,1.]],&device);
    let features=Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[1.,1.],[8.,8.]],&device);
    let weights=adjacency.clone()/adjacency.sum_dim(1);
    let messages=weights.matmul(features.clone());
    let answer=LinearConfig::new(2,3).with_bias(false).init(&device).forward(messages.clone());
    println!("Сведения после обмена: {:?}", messages.clone().into_data());
    println!("Размер после слоя: {:?}",answer.dims());
    let error=(messages.slice([3..4])-features.slice([3..4])).abs().max().into_scalar::<f32>();
    assert_eq!(answer.dims(),[4,3]); assert_eq!(error,0.0);
}
