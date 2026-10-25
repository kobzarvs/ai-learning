use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::{silu}}};

#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
 // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
 fn new(inputs:usize,width:usize,device:&Device)->Self {Self{hidden:LinearConfig::new(inputs,width).init(device),output:LinearConfig::new(width,2).init(device)}}
 fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}

fn points(n:usize,d:&Device)->(Tensor<2>,Tensor<1,Int>){
 let x=Tensor::<2>::random([n,2],Distribution::Normal(0.0,1.0),d);

 // Берём часть тензора по указанным диапазонам индексов.
 let y=(x.clone().slice([0..n,0..1])+x.clone().slice([0..n,1..2])*0.5).greater_elem(0.0).int().reshape([n]); (x,y)
}

fn fit_net(mut model:Net,x:Tensor<2>,y:Tensor<1,Int>,steps:usize,rate:f64,decay:f32)->Net {
 let mut opt=AdamWConfig::new().with_weight_decay(decay).init();

 // Ошибка классификации сравнивает оценки классов с верными метками.
 let ce=CrossEntropyLossConfig::new().init(&x.device());

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..steps {let loss=ce.forward(model.forward(x.clone()),y.clone());let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(rate,model,grads);} model
}

// Меняем форму тензора, сохраняя число и порядок элементов.
fn acc_net(model:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32 {model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()}

fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();let (x,y)=points(500,&d);let mut best:Option<Net>=None;let mut best_score=-1.0;let mut best_width=0;let mut scores=Vec::new();
 for width in [4,12,32] {d.seed(7);let model=fit_net(Net::new(2,width,&d),x.clone().slice([0..300]),y.clone().slice([0..300]),80,0.02,0.001);let score=acc_net(&model,x.clone().slice([300..400]),y.clone().slice([300..400]));scores.push(score);println!("Ширина {width}, валидация {score:.3}");if score>best_score {best_score=score;best_width=width;best=Some(model);}}
 assert!(scores.iter().all(|s|*s<=best_score));let score=acc_net(&best.unwrap(),x.slice([400..500]),y.slice([400..500]));println!("Выбрано {best_width}, валидация {best_score:.3}, тест {score:.3}");assert!(best_score>0.9 && score>0.9);
}
