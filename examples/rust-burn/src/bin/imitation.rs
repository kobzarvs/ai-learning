use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear,LinearConfig}, optim::{AdamConfig,GradientsParams}, tensor::activation::tanh};
#[derive(Module,Debug)]
struct Controller { hidden: Linear, output: Linear }
impl Controller {
    fn forward(&self,x: Tensor<2>) -> Tensor<2> { tanh(self.output.forward(tanh(self.hidden.forward(x)))) }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let x = Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&device);

    // Берём часть тензора по указанным диапазонам индексов.
    let expert = (x.clone().slice([0..400,0..1])*(-0.7)+x.clone().slice([0..400,1..2])*(-0.3)).clamp(-1.0,1.0);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = Controller {hidden:LinearConfig::new(2,16).init(&device),output:LinearConfig::new(16,1).init(&device)};

    // Читаем единственное значение тензора как обычное число Rust.
    let error = |m: &Controller| (m.forward(x.clone().slice([320..400]))-expert.clone().slice([320..400])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = error(&model);

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut opt = AdamConfig::new().init();
    for _ in 0..400 {
        let loss = (model.forward(x.clone().slice([0..320]))-expert.clone().slice([0..320])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.02,model,grads);
    }
    let after = error(&model); let mut position = 2.0_f32; let mut speed = 0.0_f32;
    for _ in 0..60 {
        let action = model.forward(Tensor::<2>::from_data([[position,speed]],&device)).into_scalar::<f32>();
        assert!(action.abs()<=1.0); speed += 0.1*action; position += 0.1*speed;
    }
    println!("Ошибка эксперта: {before} -> {after}; положение: {position}");
    assert!(after<0.025 && after<before*0.3);
    assert!(position.abs()<1.8,"Проверка относится к начальному положению 2");
}
