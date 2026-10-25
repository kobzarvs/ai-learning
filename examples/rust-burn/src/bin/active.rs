use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::{softmax}}};

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
 let d=device().autodiff();let (x,y)=points(320,&d);

 // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
 let first=fit_linear(LinearConfig::new(2,2).init(&d),x.clone().slice([0..20]),y.clone().slice([0..20]),160);

 // Отделяем тензор от графа градиентов, сохраняя его значения.
 let p=softmax(first.forward(x.clone().slice([20..240])),1).detach();
 let entropy=-(p.clone()*p.clamp_min(1e-8).log()).sum_dim(1).reshape([220]);
 let values=entropy.into_data().try_to_vec::<f32>().unwrap();
 let mut order:Vec<usize>=(0..220).collect();order.sort_by(|a,b|values[*b].total_cmp(&values[*a]));
 let chosen:Vec<i32>=order[..20].iter().map(|i|(*i+20) as i32).collect();
 assert!(chosen.iter().all(|i|*i>=20 && *i<240));
 let indices=Tensor::<1,Int>::from_ints(chosen.as_slice(),&d);

 // Случайный контроль: сортировка независимых случайных ключей без повторов.
 let keys=Tensor::<1>::random([220],Distribution::Uniform(0.0,1.0),&d).into_data().try_to_vec::<f32>().unwrap();
 let mut random:Vec<usize>=(0..220).collect();random.sort_by(|a,b|keys[*a].total_cmp(&keys[*b]));
 let ri:Vec<i32>=random[..20].iter().map(|i|(*i+20) as i32).collect();let ri=Tensor::<1,Int>::from_ints(ri.as_slice(),&d);
 let train=|ix:Tensor<1,Int>|(Tensor::cat(vec![x.clone().slice([0..20]),x.clone().select(0,ix.clone())],0),Tensor::cat(vec![y.clone().slice([0..20]),y.clone().select(0,ix)],0));
 let (ax,ay)=train(indices);let (rx,ry)=train(ri);
 let control=fit_linear(first.clone(),rx,ry,160);let active=fit_linear(first,ax,ay,160);
 let a=acc_linear(&active,x.clone().slice([240..320]),y.clone().slice([240..320]));let r=acc_linear(&control,x.slice([240..320]),y.slice([240..320]));
 println!("Запрошены {:?}; точность активного {a:.3}, случайного {r:.3}",chosen);assert!(a>0.8 && r>0.8);
}
