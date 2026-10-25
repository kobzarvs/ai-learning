use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Param,nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams}};
fn adapt(mut model:Linear,x:Tensor<2>,y:Tensor<2>)->Linear {
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.0).init();
    for _ in 0..10 {
        let loss=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.02,model,grads);
    }
    model
}
fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff(); let mut initial=LinearConfig::new(1,1).init(&device);
    for _ in 0..120 {
        let x=Tensor::<2>::random([12,1],Distribution::Normal(0.0,1.0),&device);
        let slope=Tensor::<2>::random([1,1],Distribution::Uniform(1.0,2.0),&device);
        let bias=Tensor::<2>::random([1,1],Distribution::Uniform(0.0,0.4),&device);

        // Отделяем тензор от графа градиентов, сохраняя его значения.
        let task=adapt(initial.clone(),x.clone(),(x*slope+bias).detach());

        // Регистрируем тензор как обучаемый параметр модели.
        initial.weight=Param::from_tensor((initial.weight.val()*0.85+task.weight.val()*0.15).detach());
        initial.bias=Some(Param::from_tensor((initial.bias.as_ref().unwrap().val()*0.85+task.bias.as_ref().unwrap().val()*0.15).detach()));
    }
    let x=Tensor::<2>::random([30,1],Distribution::Normal(0.0,1.0),&device); let y=x.clone()*1.7+0.15;

    // Берём часть тензора по указанным диапазонам индексов.
    let before=(initial.forward(x.clone().slice([5..30]))-y.clone().slice([5..30])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let adapted=adapt(initial,x.clone().slice([0..5]),y.clone().slice([0..5]));

    // Читаем единственное значение тензора как обычное число Rust.
    let after=(adapted.forward(x.slice([5..30]))-y.slice([5..30])).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Reptile: до {before:.6}, после 10 шагов {after:.6}");
    assert!(after<before && after<0.02);
}
