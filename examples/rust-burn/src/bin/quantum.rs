use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Qubit {angle:Param<Tensor<1>>}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();

    // Регистрируем тензор как обучаемый параметр модели.
    let mut qubit=Qubit {angle:Param::from_tensor(Tensor::full([1],0.2,&device))};

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut opt=AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..150 {
        let probability_one=(qubit.angle.val()/2.0).sin().powf_scalar(2.0);
        let loss=(probability_one-0.8).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&qubit); qubit=opt.step(0.05,qubit,grads);
    }

    // Читаем единственное значение тензора как обычное число Rust.
    let angle=qubit.angle.val().into_scalar::<f32>();
    let p=(qubit.angle.val()/2.0).sin().powf_scalar(2.0).into_scalar::<f32>();
    let p0=(angle/2.0).cos().powi(2);
    println!("Угол: {angle}; вероятность 1: {p}");
    assert!((p-0.8).abs()<0.003 && (p+p0-1.0).abs()<1e-6);
}
