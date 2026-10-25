use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear,LinearConfig}, optim::{AdamWConfig,GradientsParams},tensor::activation::silu};

#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
    fn new(inputs:usize,width:usize,outputs:usize,device:&Device)->Self {
        // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
        Self{hidden:LinearConfig::new(inputs,width).init(device),output:LinearConfig::new(width,outputs).init(device)}
    }
    fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}

fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();
    let t=Tensor::<2>::random([320,1],Distribution::Normal(0.0,1.0),&device);
    let x=Tensor::cat(vec![t.clone(),t.clone()*2.0,-t.clone(),t*0.5],1)+Tensor::<2>::random([320,4],Distribution::Normal(0.0,0.03),&device);
    let mut model=Net::new(8,24,4,&device);
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.01).init();

    // Берём часть тензора по указанным диапазонам индексов.
    let test=x.clone().slice([250..320]);
    let mask=Tensor::<2>::zeros([70,4],&device).slice_assign([0..70,1..2],Tensor::<2>::ones([70,1],&device));
    let probe=Tensor::cat(vec![test.clone()*(1.0-mask.clone()),mask],1);
    let target=test.slice([0..70,1..2]);

    // Читаем единственное значение тензора как обычное число Rust.
    let before=(model.forward(probe.clone()).slice([0..70,1..2])-target.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    for _ in 0..220 {
        let mask=Tensor::<2>::random([250,4],Distribution::Uniform(0.0,1.0),&device).lower_elem(0.35).float();
        let train=x.clone().slice([0..250]);
        let observed=train.clone()*(1.0-mask.clone());
        let prediction=model.forward(Tensor::cat(vec![observed,mask.clone()],1));
        let loss=((prediction-train).powf_scalar(2.0)*mask.clone()).sum()/mask.sum().clamp_min(1.0);

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model=optimizer.step(0.02,model,grads);
    }
    let error=(model.forward(probe).slice([0..70,1..2])-target).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE скрытого столбца: до {before:.5}, после {error:.5}");
    assert!(error<0.1 && error<before*0.25);
}
