use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct Encoder { hidden:Linear, output:Linear }
impl Encoder {
    fn forward(&self,x:Tensor<2>)->Tensor<2>{
        let z=self.output.forward(silu(self.hidden.forward(x)));
        let norm=(z.clone().powf_scalar(2.0).sum_dim(1)+1e-8).sqrt(); z/norm
    }
}
fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let x=Tensor::<2>::random([128,8],Distribution::Normal(0.0,1.0),&device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut encoder=Encoder{hidden:LinearConfig::new(8,24).init(&device),output:LinearConfig::new(24,12).init(&device)};
    let labels=Tensor::<1,Int>::arange(0..128,&device);

    // Ошибка классификации сравнивает оценки классов с верными метками.
    let loss_fn=CrossEntropyLossConfig::new().init(&device); let mut optimizer=AdamConfig::new().init();

    // Фиксированная обучающая пара для сопоставимого замера до/после.
    let probe_a=x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device);
    let probe_b=x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let before=loss_fn.forward(encoder.forward(probe_a.clone()).matmul(encoder.forward(probe_b.clone()).transpose())/0.1,labels.clone()).into_scalar::<f32>();
    for _ in 0..100 {
        let a=encoder.forward(x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device));
        let b=encoder.forward(x.clone()+Tensor::<2>::random([128,8],Distribution::Normal(0.0,0.1),&device));
        let loss=loss_fn.forward(a.matmul(b.transpose())/0.1,labels.clone());

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&encoder); encoder=optimizer.step(0.01,encoder,grads);
    }

    // Читаем единственное значение тензора как обычное число Rust.
    let after=loss_fn.forward(encoder.forward(probe_a).matmul(encoder.forward(probe_b).transpose())/0.1,labels).into_scalar::<f32>();
    println!("Контрастивная ошибка до {before:.4}, после {after:.4}");
    assert!(after<before*0.8, "Обучение должно снизить ошибку на фиксированной паре видов");
    let new=Tensor::<2>::random([40,8],Distribution::Normal(0.0,1.0),&device);
    let a=encoder.forward(new.clone());
    let b=encoder.forward(new+Tensor::<2>::random([40,8],Distribution::Normal(0.0,0.1),&device));

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let accuracy=a.matmul(b.transpose()).argmax(1).squeeze::<1>().equal(Tensor::<1,Int>::arange(0..40,&device)).float().mean().into_scalar::<f32>();
    println!("Найден свой второй вид: {accuracy:.3}"); assert!(accuracy>0.8);
}
