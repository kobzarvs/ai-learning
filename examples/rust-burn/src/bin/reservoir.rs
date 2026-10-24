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
    let w = Tensor::<2>::random([24, 24], Distribution::Normal(0.0, 0.08), &device);
    let input = Tensor::<2>::random([24, 1], Distribution::Normal(0.0, 0.3), &device);
    let mut state = Tensor::<2>::zeros([24, 1], &device);
    let mut states = Vec::new();
    for t in 0..259 {
        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        state = (w.clone().matmul(state)+input.clone()*(t as f64*0.13).sin()).tanh();
        states.push(state.clone().transpose());
    }
    let x = Tensor::cat(states, 0);
    let target: Vec<f32> = (1..260).map(|t| (t as f32*0.13).sin()).collect();
    let y = Tensor::<2>::from_data(burn::tensor::TensorData::new(target, [259, 1]), &device);

    // Подбираем выходные веса со штрафом за их величину.
    // Сингулярное разложение даёт то же решение, что (XᵀX + λI)⁻¹Xᵀy.
    let (u, s, vt) = svd::<2, 1>(x.clone().slice([20..200]), 100);
    let shrink = s.clone()/(s.powf_scalar(2.0)+0.001);
    let readout = (vt.transpose()*shrink.unsqueeze_dim::<2>(0))

        // Берём часть тензора по указанным диапазонам индексов.
        .matmul(u.transpose()).matmul(y.clone().slice([20..200]));
    let error = (x.slice([200..259]).matmul(readout)-y.slice([200..259]))

        // Читаем единственное значение тензора как обычное число Rust.
        .powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка прогноза на более позднем участке: {error}");
    assert!(error < 0.02);
}
