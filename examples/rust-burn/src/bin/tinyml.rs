use burn::{nn::{conv::Conv2dConfig, PaddingConfig2d}, tensor::{Device, Tensor}};
fn main() {
    let device = Device::flex();
    let separate = Conv2dConfig::new([16, 16], [3, 3]).with_groups(16)
        .with_bias(false).with_padding(PaddingConfig2d::Same).init(&device);
    let mix = Conv2dConfig::new([16, 32], [1, 1]).with_bias(false).init(&device);
    let x = Tensor::<4>::ones([1, 16, 24, 24], &device);
    let result = mix.forward(separate.forward(x));
    let count = separate.weight.val().dims().iter().product::<usize>() + mix.weight.val().dims().iter().product::<usize>();
    println!("Форма: {:?}; весов: {count}", result.dims());
    assert_eq!(count, 656);
}
