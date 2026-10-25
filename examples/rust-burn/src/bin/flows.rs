use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param}, optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Flow { shift: Param<Tensor<1>>, log_scale: Param<Tensor<1>> }
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let data = Tensor::<1>::random([300],Distribution::Normal(3.0,0.5),&device);

    // Регистрируем тензор как обучаемый параметр модели.
    let mut flow = Flow { shift: Param::from_tensor(Tensor::zeros([1],&device)), log_scale: Param::from_tensor(Tensor::zeros([1],&device)) };

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    for _ in 0..300 {
        let z = (data.clone()-flow.shift.val())*(-flow.log_scale.val()).exp();
        let loss = (z.powf_scalar(2.0)*0.5+flow.log_scale.val()).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &flow);
        flow = opt.step(0.05,flow,grads);
    }

    // Читаем единственное значение тензора как обычное число Rust.
    let shift = flow.shift.val().into_scalar::<f32>();
    let scale = flow.log_scale.val().exp().into_scalar::<f32>();
    let z = Tensor::<1>::random([5],Distribution::Normal(0.0,1.0),&device);
    let samples = z.clone()*scale+shift;
    let error = ((samples-shift)/scale-z).abs().max().into_scalar::<f32>();
    println!("Сдвиг: {shift}; масштаб: {scale}; ошибка обратимости: {error}");
    assert!((shift-3.0).abs()<0.2 && (scale-0.5).abs()<0.15 && error<1e-5);
}
