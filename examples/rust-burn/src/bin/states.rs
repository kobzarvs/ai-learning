use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn normal(n: usize, device: &Device) -> Vec<f32> {
    Tensor::<1>::random([n], Distribution::Normal(0.0, 1.0), device)
        .into_data().try_to_vec::<f32>().unwrap()
}

fn main() {
    let device=device(); let noise=normal(60,&device);
    let truth: Vec<f32>=(0..60).map(|i| 2.0*i as f32/59.0).collect();
    let observed: Vec<f32>=truth.iter().zip(noise).map(|(&t,n)| t+0.35*n).collect();
    let mut mean=0.0_f32; let mut variance=1.0_f32; let mut estimates=Vec::new();
    for &measurement in &observed {
        variance+=0.01; let predicted_variance=variance;
        let gain=variance/(variance+0.35_f32.powi(2));
        mean+=gain*(measurement-mean); variance*=1.0-gain; estimates.push(mean);
        assert!(gain>0.0 && gain<1.0 && variance>0.0 && variance<predicted_variance);
    }
    let target=Tensor::<1>::from_data(burn::tensor::TensorData::new(truth,[60]),&device);

    // Читаем единственное значение тензора как обычное число Rust.
    let raw=(Tensor::<1>::from_data(burn::tensor::TensorData::new(observed,[60]),&device)-target.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let filtered=(Tensor::<1>::from_data(burn::tensor::TensorData::new(estimates,[60]),&device)-target).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE измерений {raw}; фильтра {filtered}"); assert!(filtered<raw && filtered<0.08);
}
