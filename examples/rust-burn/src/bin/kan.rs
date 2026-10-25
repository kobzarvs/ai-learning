use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Edges { coefficients:Param<Tensor<2>> }
impl Edges {
    fn forward(&self,x:Tensor<2>,centers:Tensor<3>)->Tensor<2>{
        let n=x.dims()[0];

        // Меняем форму тензора, сохраняя число и порядок элементов.
        let bases=((x.reshape([n,2,1])-centers)/0.25).powf_scalar(2.0).neg().exp();
        (bases*self.coefficients.val().reshape([1,2,12])).sum_dim(2).sum_dim(1).reshape([n,1])
    }
}
fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let x=Tensor::<2>::random([400,2],Distribution::Uniform(-1.0,1.0),&device);

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let y=(x.clone().slice([0..400,0..1]).sin()+x.clone().slice([0..400,1..2]).powf_scalar(2.0)).detach();
    let centers=Tensor::<1>::from_floats((0..12).map(|i|-1.0+2.0*i as f32/11.0).collect::<Vec<_>>().as_slice(),&device).reshape([1,1,12]);

    // Регистрируем тензор как обучаемый параметр модели.
    let mut model=Edges{coefficients:Param::from_tensor(Tensor::<2>::zeros([2,12],&device))};

    // Берём часть тензора по указанным диапазонам индексов.
    let tx=x.clone().slice([320..400]); let ty=y.clone().slice([320..400]);

    // Читаем единственное значение тензора как обычное число Rust.
    let before=ty.clone().powf_scalar(2.0).mean().into_scalar::<f32>();

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer=AdamConfig::new().init();
    for _ in 0..300 {
        let loss=(model.forward(x.clone().slice([0..320]),centers.clone())-y.clone().slice([0..320])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.03,model,grads);
    }
    let error=(model.forward(tx,centers)-ty).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE до {before:.6}, после {error:.6}"); assert!(error<0.01 && error<before*0.1);
}
