use burn::tensor::{Device, Tensor, activation::softmax};
fn main() {
    let device = Device::flex();
    let all_keys = Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[1.,1.]], &device);
    let all_values = Tensor::<2>::from_data([[2.0_f32],[4.],[8.]], &device);
    for length in 1..=3 {
        let keys = all_keys.clone().slice([0..length, 0..2]);
        let values = all_values.clone().slice([0..length, 0..1]);
        let query = Tensor::<2>::from_data([[1.0_f32,1.]], &device);
        let scores = softmax(query.matmul(keys.transpose()) / (2.0_f64).sqrt(), 1);
        let answer = scores.matmul(values).into_scalar::<f32>();
        println!("Доступно позиций: {length}; ответ: {answer:.4}");
    }
}
