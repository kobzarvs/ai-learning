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
    let device = device().autodiff();
    let x=Tensor::<2>::random([240,2], Distribution::Normal(0.0,1.0), &device);

    // Берём часть тензора по указанным диапазонам индексов.
    let labels=(x.clone().slice([0..240,0..1])+x.clone().slice([0..240,1..2])*0.5)
        .greater_elem(0.0).float();
    let y=labels.clone()*2.0-1.0;

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model=LinearConfig::new(2,1).init(&device);

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer=AdamConfig::new().init();
    let objective = |score: Tensor<2>, weight: Tensor<2>| {
        (y.clone().slice([0..180])*score*(-1.0)+1.0).clamp_min(0.0).mean()
            +weight.powf_scalar(2.0).sum()*0.01
    };

    // Читаем единственное значение тензора как обычное число Rust.
    let before=objective(model.forward(x.clone().slice([0..180])),model.weight.val()).into_scalar::<f32>();
    for _ in 0..250 {
        let loss=objective(model.forward(x.clone().slice([0..180])),model.weight.val());

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(), &model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model=optimizer.step(0.03,model,grads);
    }
    let after=objective(model.forward(x.clone().slice([0..180])),model.weight.val()).into_scalar::<f32>();
    let accuracy=model.forward(x.slice([180..240])).greater_elem(0.0)
        .equal(labels.slice([180..240]).greater_elem(0.5)).float().mean().into_scalar::<f32>();
    println!("Hinge + L2: {before} → {after}; новые строки: {accuracy}");
    assert!(after<before && after<0.45 && accuracy>0.9);
}
