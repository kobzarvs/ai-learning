use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{conv::Conv2dConfig, PaddingConfig2d}};
fn main() {
    let device = device();
    let regular = Conv2dConfig::new([16,32],[3,3]).with_padding(PaddingConfig2d::Same).with_bias(false).init(&device);
    let depthwise = Conv2dConfig::new([16,16],[3,3]).with_groups(16).with_padding(PaddingConfig2d::Same).with_bias(false).init(&device);
    let pointwise = Conv2dConfig::new([16,32],[1,1]).with_bias(false).init(&device);
    let x = Tensor::<4>::random([1,16,24,24], Distribution::Normal(0.0,1.0), &device);
    let full = regular.forward(x.clone());
    let compact = pointwise.forward(depthwise.forward(x));
    let a=regular.num_params(); let b=depthwise.num_params()+pointwise.num_params();
    println!("Обычная: {a} весов, {} байт, {:?}", a*4, full.dims());
    println!("Компактная: {b} весов, {} байт, {:?}", b*4, compact.dims());
    assert_eq!((a,b),(4608,656)); assert_eq!(full.dims(),compact.dims());
}
