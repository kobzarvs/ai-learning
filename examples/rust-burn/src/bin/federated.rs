use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Param,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::Int};

fn points(n:usize,d:&Device)->(Tensor<2>,Tensor<1,Int>){
 let x=Tensor::<2>::random([n,2],Distribution::Normal(0.0,1.0),d);

 // Берём часть тензора по указанным диапазонам индексов.
 let y=(x.clone().slice([0..n,0..1])+x.clone().slice([0..n,1..2])*0.5).greater_elem(0.0).int().reshape([n]); (x,y)
}

fn fit_linear(mut model:Linear,x:Tensor<2>,y:Tensor<1,Int>,steps:usize)->Linear {
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();

 // Ошибка классификации сравнивает оценки классов с верными метками.
 let ce=CrossEntropyLossConfig::new().init(&x.device());

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..steps {let loss=ce.forward(model.forward(x.clone()),y.clone());let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);} model
}

// Меняем форму тензора, сохраняя число и порядок элементов.
fn acc_linear(model:&Linear,x:Tensor<2>,y:Tensor<1,Int>)->f32 {model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()}

fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();let (x,y)=points(400,&d);let mut global=LinearConfig::new(2,2).init(&d);
 let before=acc_linear(&global,x.clone().slice([300..400]),y.clone().slice([300..400]));
 for _ in 0..15 {
 let a=fit_linear(global.clone(),x.clone().slice([0..150]),y.clone().slice([0..150]),5);
 let b=fit_linear(global.clone(),x.clone().slice([150..300]),y.clone().slice([150..300]),5);
 let w=(a.weight.val()+b.weight.val())*0.5;let bias=(a.bias.as_ref().unwrap().val()+b.bias.as_ref().unwrap().val())*0.5;

 // Регистрируем тензор как обучаемый параметр модели.
 global.weight=Param::from_tensor(w.clone().detach());global.bias=Some(Param::from_tensor(bias.clone().detach()));

 // Читаем единственное значение тензора как обычное число Rust.
 assert!((global.weight.val()-w).abs().max().into_scalar::<f32>()<1e-6);
 }
 let score=acc_linear(&global,x.slice([300..400]),y.slice([300..400]));println!("Общая модель: до {before:.3}, после {score:.3}");assert!(score>0.9);
}
