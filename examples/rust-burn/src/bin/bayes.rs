use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device=device();
    let observations=[1_usize,1,0,1,1,0,1,1];
    let a=1+observations.iter().sum::<usize>();
    let b=1+observations.len()-observations.iter().sum::<usize>();

    // Для целого shape Gamma(k,1) — сумма k независимых Exp(1).
    // X/(X+Y), X~Gamma(a,1), Y~Gamma(b,1), имеет Beta(a,b).
    let gamma_a=Tensor::<2>::random([10000,a],Distribution::Uniform(0.0,1.0),&device)
        .clamp_min(1e-7).log().sum_dim(1)*(-1.0);
    let gamma_b=Tensor::<2>::random([10000,b],Distribution::Uniform(0.0,1.0),&device)
        .clamp_min(1e-7).log().sum_dim(1)*(-1.0);
    let samples=gamma_a.clone()/(gamma_a+gamma_b);

    // Читаем единственное значение тензора как обычное число Rust.
    let empirical=samples.clone().mean().into_scalar::<f32>();
    let mut values=samples.into_data().try_to_vec::<f32>().unwrap(); values.sort_by(f32::total_cmp);
    let lower=values[250]; let upper=values[9749]; let mean=a as f32/(a+b) as f32;
    println!("Beta({a},{b}); среднее {mean}; выборочное {empirical}; 95% [{lower}, {upper}]");
    assert_eq!((a,b),(7,3)); assert!((mean-0.7).abs()<1e-6);
    assert!((empirical-mean).abs()<0.015 && lower>0.35 && lower<0.5 && upper>0.9 && upper<0.98);
}
