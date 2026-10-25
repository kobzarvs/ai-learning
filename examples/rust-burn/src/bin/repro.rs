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

fn run(seed:u64,x:Tensor<2>,y:Tensor<2>,device:&Device)->Tensor<2>{
    device.seed(seed);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let model=fit(LinearConfig::new(2,1).init(device),x.clone(),y,20);

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    model.forward(x).detach()
}
fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let x=Tensor::<2>::random([100,2],Distribution::Normal(0.0,1.0),&device);
    let y=x.clone().sum_dim(1);
    let a=run(7,x.clone(),y.clone(),&device);
    let b=run(7,x.clone(),y.clone(),&device);
    let c=run(8,x,y,&device);

    // Читаем единственное значение тензора как обычное число Rust.
    let same=(a.clone()-b).abs().max().into_scalar::<f32>();
    let different=(a-c).abs().max().into_scalar::<f32>();
    println!("Burn 0.22.1; одинаковый seed: max diff {same}; другой seed: {different}");
    assert_eq!(same,0.0); assert!(different>1e-4);
}
