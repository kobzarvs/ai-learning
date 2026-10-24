use burn::tensor::{Device, Tensor, activation::softmax};
fn main() {
    let device = Device::flex();
    let query = Tensor::<2>::from_data([[1.0_f32, 0.], [0., 1.]], &device);
    let keys = Tensor::<2>::from_data([[1.0_f32, 0.], [0., 1.], [1., 1.]], &device);
    let values = Tensor::<2>::from_data([[10.0_f32], [20.], [30.]], &device);
    let scores = query.matmul(keys.transpose()) / (2.0_f64).sqrt();
    let attention = softmax(scores, 1);
    let answer = attention.clone().matmul(values);
    println!("Веса внимания: {:?}", attention.into_data().try_to_vec::<f32>().unwrap());
    println!("Ответы: {:?}", answer.into_data().try_to_vec::<f32>().unwrap());
}
