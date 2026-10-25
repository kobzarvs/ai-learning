use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param}, optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Energy { center: Param<Tensor<1>> }
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let data = Tensor::<1>::random([200],Distribution::Normal(1.5,0.2),&device);

    // Регистрируем тензор как обучаемый параметр модели.
    let mut model = Energy { center: Param::from_tensor(Tensor::zeros([1],&device)) };

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..100 {
        let loss = (data.clone()-model.center.val()).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.05,model,grads);
    }

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let center = model.center.val().detach();

    // Читаем единственное значение тензора как обычное число Rust.
    let learned = center.clone().into_scalar::<f32>();
    let mut x = Tensor::<1>::random([100],Distribution::Normal(0.0,1.0),&device);
    for _ in 0..80 {
        // Просим сохранять градиент по этому тензору.
        x = x.detach().require_grad();
        let energy = (x.clone()-center.clone()).powf_scalar(2.0)*0.5;

        // Обратный проход вычисляет градиенты ошибки.
        let grads = energy.sum().backward();
        let gradient = x.grad(&grads).unwrap();
        x = (x-gradient*0.1+Tensor::<1>::random([100],Distribution::Normal(0.0,0.02_f64.sqrt()),&device)).detach();
    }
    let mean = x.clone().mean().into_scalar::<f32>();
    let std = (x-mean).powf_scalar(2.0).mean().sqrt().into_scalar::<f32>();
    println!("Центр: {learned}; среднее образцов: {mean}; разброс: {std}");
    assert!((learned-1.5).abs()<0.12 && (mean-learned).abs()<0.15 && std>0.15 && std<0.5);
}
