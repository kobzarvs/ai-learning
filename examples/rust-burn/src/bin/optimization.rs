use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex().autodiff();
    let mut value = Tensor::<1>::from_data([5.0_f32], &device).require_grad();
    for _ in 0..80 {
        let loss = (value.clone()-2.0).powf_scalar(2.0).mean();
        let gradients = loss.backward();
        let gradient = value.grad(&gradients).unwrap();
        // detach отделяет новый шаг от графа предыдущих шагов.
        value = (value-gradient*0.1).detach().require_grad();
    }
    let answer = value.into_scalar::<f32>();
    println!("Параметр после обучения: {answer}");
    assert!((answer-2.0).abs() < 1e-3);
}
