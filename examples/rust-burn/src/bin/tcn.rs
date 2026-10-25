use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{conv::{Conv1d, Conv1dConfig}, Linear, LinearConfig, loss::CrossEntropyLossConfig}, optim::{AdamConfig, GradientsParams}, tensor::{Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { conv: Conv1d, output: Linear }
impl Model {
    fn forward(&self,x:Tensor<3>) -> Tensor<2> {
        self.output.forward(silu(self.conv.forward(x)).mean_dim(2).squeeze::<2>())
    }
}
fn accuracy(model:&Model,x:Tensor<3>,y:Tensor<1,Int>) -> f32 {
    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    model.forward(x).argmax(1).squeeze::<1>().equal(y).float().mean().into_scalar()
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let y=Tensor::<1>::random([300],Distribution::Uniform(0.0,1.0),&device).greater_elem(0.5).int();

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let frequency=y.clone().float().reshape([300,1,1])*6.0+3.0;
    let t=Tensor::<1>::from_floats((0..64).map(|i|i as f32/63.0).collect::<Vec<_>>().as_slice(),&device).reshape([1,1,64]);
    let x=(frequency*t*(2.0*std::f32::consts::PI)).sin()+Tensor::<3>::random([300,1,64],Distribution::Normal(0.0,0.15),&device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model=Model{conv:Conv1dConfig::new(1,12,5).with_dilation(2).init(&device),output:LinearConfig::new(12,2).init(&device)};

    // Ошибка классификации сравнивает оценки классов с верными метками.
    let mut optimizer=AdamConfig::new().init(); let loss_fn=CrossEntropyLossConfig::new().init(&device);

    // Берём часть тензора по указанным диапазонам индексов.
    let before=accuracy(&model,x.clone().slice([240..300]),y.clone().slice([240..300]));
    for _ in 0..180 {
        let loss=loss_fn.forward(model.forward(x.clone().slice([0..240])),y.clone().slice([0..240]));

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.02,model,grads);
    }
    let after=accuracy(&model,x.slice([240..300]),y.slice([240..300]));
    println!("Точность до {before:.3}, после {after:.3}"); assert!(after>0.9);
}
