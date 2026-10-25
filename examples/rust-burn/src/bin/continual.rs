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

fn fit_net(mut model:Net,x:Tensor<2>,y:Tensor<1,Int>,steps:usize)->Net {
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();

 // Ошибка классификации сравнивает оценки классов с верными метками.
 let ce=CrossEntropyLossConfig::new().init(&x.device());

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..steps {let loss=ce.forward(model.forward(x.clone()),y.clone());let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);} model
}

// Меняем форму тензора, сохраняя число и порядок элементов.
fn acc_net(model:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32 {model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()}

fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();let x=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);

 // Берём часть тензора по указанным диапазонам индексов.
 let ya=x.clone().slice([0..400,0..1]).greater_elem(0.0).int().reshape([400]);let yb=x.clone().slice([0..400,1..2]).greater_elem(0.0).int().reshape([400]);
 let a=Tensor::cat(vec![x.clone(),Tensor::zeros([400,1],&d)],1);let b=Tensor::cat(vec![x,Tensor::ones([400,1],&d)],1);
 let first=fit_net(Net::new(3,24,&d),a.clone().slice([0..300]),ya.clone().slice([0..300]),160);
 let before=acc_net(&first,a.clone().slice([300..400]),ya.clone().slice([300..400]));
 let naive=fit_net(first.clone(),b.clone().slice([0..300]),yb.clone().slice([0..300]),160);
 let replay=fit_net(first,Tensor::cat(vec![b.clone().slice([0..300]),a.clone().slice([0..80])],0),Tensor::cat(vec![yb.clone().slice([0..300]),ya.clone().slice([0..80])],0),160);
 let old_naive=acc_net(&naive,a.clone().slice([300..400]),ya.clone().slice([300..400]));let old=acc_net(&replay,a.slice([300..400]),ya.slice([300..400]));
 let new_naive=acc_net(&naive,b.clone().slice([300..400]),yb.clone().slice([300..400]));let new=acc_net(&replay,b.slice([300..400]),yb.slice([300..400]));
 println!("Старая до {before:.3}, после {old_naive:.3}, с памятью {old:.3}; новая {new_naive:.3}, с памятью {new:.3}");
 assert!(before>0.9 && old>0.85 && new>0.85 && new_naive>0.9);assert!(old>old_naive+0.1);
}
