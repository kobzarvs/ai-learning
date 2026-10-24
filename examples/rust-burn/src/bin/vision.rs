use burn::{nn::{conv::Conv2dConfig, PaddingConfig2d}, tensor::{Device, Tensor}};
fn main() {
    let device = Device::flex();
    let image = Tensor::<4>::ones([2, 1, 8, 8], &device);
    let convolution = Conv2dConfig::new([1, 8], [3, 3])
        .with_padding(PaddingConfig2d::Same).init(&device);
    let features = convolution.forward(image);
    println!("После свёртки: {:?}", features.dims());
    assert_eq!(features.dims(), [2, 8, 8, 8]);
}
