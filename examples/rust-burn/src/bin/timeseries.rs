use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams}};
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let time=Tensor::<1>::from_data(burn::tensor::TensorData::new((0..300).map(|i| i as f32*0.1).collect::<Vec<_>>(),[300]),&device);
    let signal=time.sin()+Tensor::<1>::random([300],Distribution::Normal(0.0,0.05),&device);

    // Берём часть тензора по указанным диапазонам индексов.
    let columns: Vec<Tensor<2>>=(0..6).map(|offset| signal.clone().slice([offset..offset+294]).unsqueeze_dim::<2>(1)).collect();
    let x=Tensor::cat(columns,1); let y=signal.slice([6..300]).unsqueeze_dim::<2>(1);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model=LinearConfig::new(6,1).init(&device); let mut optimizer=AdamConfig::new().init();

    // Читаем единственное значение тензора как обычное число Rust.
    let initial=(model.forward(x.clone().slice([220..294]))-y.clone().slice([220..294])).powf_scalar(2.0).mean().into_scalar::<f32>();
    for _ in 0..200 {
        let loss=(model.forward(x.clone().slice([0..220]))-y.clone().slice([0..220])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.02,model,grads);
    }
    let error=(model.forward(x.clone().slice([220..294]))-y.clone().slice([220..294])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let baseline=(x.slice([220..294,5..6])-y.slice([220..294])).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка до {initial}; после {error}; повтор последнего {baseline}");
    assert!(error<initial && error<0.02);

    // Сравнение с последним значением измеряется, а не гарантируется любым
    // шумом/стартом.
}
