use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::linalg::svd;
fn main() {
    let device = device();
    let weights = Tensor::<2>::random([64, 4], Distribution::Normal(0.0, 1.0), &device)

        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        .matmul(Tensor::<2>::random([4, 32], Distribution::Normal(0.0, 1.0), &device))
        + Tensor::<2>::random([64, 32], Distribution::Normal(0.0, 0.01), &device);
    let (u, s, vt) = svd::<2, 1>(weights.clone(), 100);

    // Берём часть тензора по указанным диапазонам индексов.
    let left = u.slice([0..64, 0..4])*s.slice([0..4]).unsqueeze_dim::<2>(0);
    let right = vt.slice([0..4, 0..32]);
    let x = Tensor::<2>::random([20, 32], Distribution::Normal(0.0, 1.0), &device);
    let error = (x.clone().matmul(right.transpose()).matmul(left.transpose())

        // Читаем единственное значение тензора как обычное число Rust.
        - x.matmul(weights.transpose())).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка: {error}; параметров до: 2048, после: 384");
    assert!(error < 0.02);
}
