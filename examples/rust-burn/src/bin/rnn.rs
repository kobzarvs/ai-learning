use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Lstm, LstmConfig, Linear, LinearConfig}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Model { memory: Lstm, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let n = x.dims()[0];
        let (states, _) = self.memory.forward(x, None);

        // Берём часть тензора по указанным диапазонам индексов.
        self.output.forward(states.slice([0..n, 11..12, 0..16]).reshape([n,16]))
    }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let x = Tensor::<3>::random([300,12,1], Distribution::Normal(0.0,1.0), &device);

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let y = x.clone().mean_dim(1).reshape([300,1]).detach();

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = Model { memory: LstmConfig::new(1,16,true).init(&device), output: LinearConfig::new(16,1).init(&device) };
    let tx = x.clone().slice([240..300]); let ty = y.clone().slice([240..300]);

    // Читаем единственное значение тензора как обычное число Rust.
    let baseline = ty.clone().powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = (model.forward(tx.clone())-ty.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..260 {
        let loss = (model.forward(x.clone().slice([0..240]))-y.clone().slice([0..240])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model = optimizer.step(0.01,model,grads);
    }
    let error = (model.forward(tx)-ty).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE до {before:.5}, после {error:.5}, нулевой прогноз {baseline:.5}");
    assert!(error < baseline*0.5 && error < before*0.5);
}
