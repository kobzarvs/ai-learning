use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::linalg::svd;
fn main() {
    let device=device();
    let t=Tensor::<2>::from_data(burn::tensor::TensorData::new((0..120).map(|i| -2.0+4.0*i as f32/119.0).collect::<Vec<_>>(),[120,1]),&device);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let x=t.matmul(Tensor::<2>::from_data([[1.0_f32,2.0,-1.0]],&device))
        +Tensor::<2>::random([120,3],Distribution::Normal(0.0,0.05),&device);

    // Берём часть тензора по указанным диапазонам индексов.
    let train=x.clone().slice([0..90]); let mean=train.clone().mean_dim(0);
    let (_,s,vt)=svd::<2,1>(train-mean.clone(),100);
    let direction=vt.slice([0..1,0..3]); let test=x.slice([90..120]);
    let compressed=(test.clone()-mean.clone()).matmul(direction.clone().transpose());
    assert_eq!(compressed.dims(),[30,1]);
    let restored=compressed.matmul(direction.clone())+mean;

    // Читаем единственное значение тензора как обычное число Rust.
    let error=(restored-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    let unit=direction.powf_scalar(2.0).sum().into_scalar::<f32>();
    println!("[30,3] → [30,1]; MSE {error}; сингулярные числа {:?}",s.into_data());
    assert!(error<0.01 && (unit-1.0).abs()<1e-3);
}
