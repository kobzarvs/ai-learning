use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let x = Tensor::<2>::from_data([[1.0_f32, 2., 3.], [4., 5., 6.]], &device);
    let w = Tensor::<2>::from_data([[1.0_f32, 0.], [0., 1.], [1., 1.]], &device);
    let output = x.matmul(w);
    println!("Форма: {:?}", output.dims());
    let values = output.into_data().try_to_vec::<f32>().unwrap();
    println!("Значения: {values:?}");
    assert_eq!(values, vec![4., 5., 10., 11.]);
}
