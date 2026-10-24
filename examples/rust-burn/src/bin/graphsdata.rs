use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let links = Tensor::<2>::from_data([[0.5_f32,0.5,0.],[0.5,0.5,0.],[0.,0.,1.]], &device);
    let features = Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[8.,8.]], &device);
    let shared = links.matmul(features).into_data().try_to_vec::<f32>().unwrap();
    println!("Признаки после обмена: {shared:?}");
    assert_eq!(shared, vec![0.5,0.5,0.5,0.5,8.,8.]);
}
