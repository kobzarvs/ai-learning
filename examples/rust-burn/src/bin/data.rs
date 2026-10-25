use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main(){
 let d=device();let raw=Tensor::<2>::random([300,3],Distribution::Normal(0.0,1.0),&d);

 // Берём часть тензора по указанным диапазонам индексов.
 let x=Tensor::cat(vec![raw.clone().slice([0..240]),raw.slice([240..300])+1.5],0);
 let train=x.clone().slice([0..180]);let validation=x.clone().slice([180..240]);let test=x.clone().slice([240..300]);
 let mean=train.clone().mean_dim(0);let centered=train.clone()-mean.clone();

 // torch.std использует поправку Бесселя: деление на n-1.
 let std=(centered.clone().powf_scalar(2.0).sum_dim(0)/179.0).sqrt().clamp_min(1e-6);
 let train_scaled=centered/std.clone();let test_scaled=(test.clone()-mean.clone())/std.clone();
 let leaked_mean=x.clone().mean_dim(0);let leaked_train=(train-leaked_mean.clone())/std.clone();
 println!("Формы {:?}/{:?}/{:?}; средние train {:?}, test {:?}; train при утечке {:?}",train_scaled.dims(),validation.dims(),test.dims(),train_scaled.clone().mean_dim(0).into_data(),test_scaled.clone().mean_dim(0).into_data(),leaked_train.clone().mean_dim(0).into_data());

 // Читаем единственное значение тензора как обычное число Rust.
 assert!(train_scaled.mean_dim(0).abs().max().into_scalar::<f32>()<1e-5);
 assert!((leaked_mean-mean).abs().max().into_scalar::<f32>()>0.1);
 assert!(test_scaled.mean().into_scalar::<f32>()>0.5);
 assert!(leaked_train.mean_dim(0).abs().max().into_scalar::<f32>()>0.1);
}
