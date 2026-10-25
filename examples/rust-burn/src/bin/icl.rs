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

fn tasks(n:usize,device:&Device)->(Tensor<2>,Tensor<2>){
    let a=Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device);
    let b=Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device);
    let query=Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device);
    let one=Tensor::<2>::ones([n,1],device);

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    (Tensor::cat(vec![-one.clone(),-a.clone()+b.clone(),one,a.clone()+b.clone(),query.clone()],1),(a*query+b).detach())
}
fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff(); let mut model=Net::new(5,64,1,&device);
    let (probe,target)=tasks(100,&device);

    // Читаем единственное значение тензора как обычное число Rust.
    let before=(model.forward(probe.clone())-target.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.01).init();
    for _ in 0..500 {
        let (x,y)=tasks(128,&device); let loss=(model.forward(x)-y).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model);model=optimizer.step(0.01,model,grads);
    }

    // Новые правила после обучения; никаких шагов optimizer на тесте.
    let (x,y)=tasks(100,&device);
    let error=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let no_context=x.slice_assign([0..100,1..2],Tensor::<2>::zeros([100,1],&device)).slice_assign([0..100,3..4],Tensor::<2>::zeros([100,1],&device));
    let ablated=(model.forward(no_context)-y).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("До обучения {before:.5}; новые правила {error:.5}; ответы контекста обнулены {ablated:.5}");
    assert!(error<0.02 && error<before*0.1 && ablated>error*5.0);
}
