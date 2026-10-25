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
 let d=device().autodiff();let (x,y)=points(400,&d);
 let margin=(x.clone().slice([0..300,0..1])+x.clone().slice([0..300,1..2])*0.5).abs().reshape([300]).into_data().try_to_vec::<f32>().unwrap();
 let mut order:Vec<usize>=(0..300).collect();order.sort_by(|a,b|margin[*b].total_cmp(&margin[*a]));assert!(margin[order[0]]>=margin[order[299]]);
 let initial=Net::new(2,12,&d);let mut model=initial.clone();
 for size in [50,120,300] {let chosen:Vec<i32>=order[..size].iter().map(|i|*i as i32).collect();let ix=Tensor::<1,Int>::from_ints(chosen.as_slice(),&d);model=fit_net(model,x.clone().select(0,ix.clone()),y.clone().select(0,ix),50);println!("Объектов {size}: {:.3}",acc_net(&model,x.clone().slice([300..400]),y.clone().slice([300..400])));}

 // Те же три сброса AdamW и 150 обновлений, но все 300 точек с начала.
 let mut control=initial;for _ in 0..3 {control=fit_net(control,x.clone().slice([0..300]),y.clone().slice([0..300]),50);}
 let score=acc_net(&model,x.clone().slice([300..400]),y.clone().slice([300..400]));let baseline=acc_net(&control,x.slice([300..400]),y.slice([300..400]));println!("Учебный план {score:.3}, все сразу {baseline:.3}");assert!(score>0.9 && baseline>0.9);
}
