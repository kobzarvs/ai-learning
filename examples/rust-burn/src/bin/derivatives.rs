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

    // Просим сохранять градиент по этому тензору.
    let x = Tensor::<1>::from_data([3.0_f32], &device).require_grad();
    let y = x.clone().powf_scalar(2.0)+x.clone()*2.0;

    // Обратный проход вычисляет градиенты ошибки.
    let grads = y.clone().backward();

    // Читаем единственное значение тензора как обычное число Rust.
    let derivative = x.grad(&grads).unwrap().into_scalar::<f32>();
    let h = 0.001;

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let moved = x.detach()+h;
    let numerical = ((moved.clone().powf_scalar(2.0)+moved*2.0-y.detach())/h).into_scalar::<f32>();
    println!("Производная Burn: {derivative}; численная оценка: {numerical}");
    assert!((derivative-8.0).abs()<1e-6 && (numerical-8.0).abs()<0.01);
}
