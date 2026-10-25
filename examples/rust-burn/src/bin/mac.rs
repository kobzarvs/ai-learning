use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams}};
fn fit(mut model:Linear,x:Tensor<2>,y:Tensor<2>,steps:usize)->Linear {
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.0).init();
    for _ in 0..steps {
        let loss=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model=optimizer.step(0.02,model,grads);
    }
    model
}

fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let x=Tensor::<2>::random([64,2],Distribution::Normal(0.0,1.0),&device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let y=x.clone().sum_dim(1); let model=LinearConfig::new(2,1).init(&device);

    // Читаем единственное значение тензора как обычное число Rust.
    let before=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let model=fit(model,x.clone(),y.clone(),150);
    let after=(model.forward(x)-y).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Выбранное устройство {:?}; MSE до {before:.6}, после {after:.6}",device);
    assert!(after<0.01 && after<before*0.02);
}
