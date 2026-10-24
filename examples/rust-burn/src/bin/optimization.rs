use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    for rate in [0.05, 0.4, 1.1] {
        // Просим сохранять градиент по этому тензору.
        let mut parameter = Tensor::<1>::from_data([5.0_f32], &device).require_grad();
        let mut losses = Vec::new();
        for _ in 0..20 {
            let loss = (parameter.clone()-2.0).powf_scalar(2.0).mean();
            // Читаем единственное значение тензора как обычное число Rust.
            losses.push(loss.clone().into_scalar::<f32>());
            // Обратный проход вычисляет градиенты ошибки.
            let grads = loss.backward();
            // Отделяем тензор от графа градиентов, сохраняя его значения.
            parameter = (parameter.clone()-parameter.grad(&grads).unwrap()*rate).detach().require_grad();
        }
        println!("Шаг: {rate}; первые ошибки: {:?}; последняя: {}", &losses[..3], losses[19]);
        assert!(if rate<1.0 { losses[19]<losses[0] } else { losses[19]>losses[0] });
    }
}
