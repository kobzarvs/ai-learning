use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device=device();
    let x=Tensor::cat(vec![Tensor::<2>::random([120,1],Distribution::Normal(-1.5,0.3),&device),Tensor::<2>::random([120,1],Distribution::Normal(1.5,0.3),&device)],0);
    let mut means=Tensor::<2>::from_data([[-1.0_f32,1.0]],&device);
    let mut variance=Tensor::<2>::ones([1,2],&device); let mut weights=Tensor::<2>::ones([1,2],&device)*0.5;
    let mut previous=f32::NEG_INFINITY; let mut first=0.0;
    for iteration in 0..30 {
        let score=((x.clone()-means.clone()).powf_scalar(2.0)/variance.clone()+variance.clone().log())*(-0.5)+weights.clone().log();

        // Стабильная log-sum-exp для проверки монотонности EM.
        let max=score.clone().max_dim(1);

        // Читаем единственное значение тензора как обычное число Rust.
        let likelihood=((score.clone()-max.clone()).exp().sum_dim(1).log()+max).mean().into_scalar::<f32>();
        if iteration==0 {first=likelihood;}
        assert!(likelihood+1e-4>=previous); previous=likelihood;

        // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
        let responsibility=burn::tensor::activation::softmax(score,1);
        let row_error=(responsibility.clone().sum_dim(1)-1.0).abs().max().into_scalar::<f32>(); assert!(row_error<1e-5);
        let count=responsibility.clone().sum_dim(0);
        means=(responsibility.clone()*x.clone()).sum_dim(0)/count.clone();
        variance=(responsibility*(x.clone()-means.clone()).powf_scalar(2.0)).sum_dim(0)/count.clone();
        variance=variance.clamp_min(1e-4); weights=count/240.0;
    }
    let mu=means.into_data().try_to_vec::<f32>().unwrap(); let pi=weights.into_data().try_to_vec::<f32>().unwrap();
    println!("Центры {mu:?}; доли {pi:?}; log L {first} → {previous}");
    assert!((mu[0]+1.5).abs()<0.15 && (mu[1]-1.5).abs()<0.15);
    assert!(pi.iter().all(|&p| (p-0.5).abs()<0.05) && previous>first+0.5);
}
