use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::Int};
fn permutation(n:usize,d:&Device)->Tensor<1,Int>{
 let keys=Tensor::<1>::random([n],Distribution::Uniform(0.0,1.0),d).into_data().try_to_vec::<f32>().unwrap();
 let mut ix:Vec<usize>=(0..n).collect();ix.sort_by(|a,b|keys[*a].total_cmp(&keys[*b]));
 let ix:Vec<i32>=ix.into_iter().map(|i|i as i32).collect();Tensor::from_ints(ix.as_slice(),d)
}

// Меняем форму тензора, сохраняя число и порядок элементов.
fn accuracy(m:&Linear,x:Tensor<2>,y:Tensor<1,Int>)->f32 {let n=y.dims()[0];m.forward(x).argmax(1).reshape([n]).equal(y).float().mean().into_scalar()}

// Включаем запись вычислений, чтобы затем получить градиенты ошибки.
fn main(){let d=device().autodiff();let x=Tensor::<2>::random([400,3],Distribution::Normal(0.0,1.0),&d);let y=(x.clone().slice([0..400,0..1])*2.0+x.clone().slice([0..400,1..2])).greater_elem(0.0).int().reshape([400]);

 // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
 let mut m=LinearConfig::new(3,2).init(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();let ce=CrossEntropyLossConfig::new().init(&d);

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..160 {let loss=ce.forward(m.forward(x.clone().slice([0..300])),y.clone().slice([0..300]));let g=GradientsParams::from_grads(loss.backward(),&m);m=opt.step(0.02,m,g);}

 // Берём часть тензора по указанным диапазонам индексов.
 let test=x.slice([300..400]);let labels=y.slice([300..400]);let baseline=accuracy(&m,test.clone(),labels.clone());let mut drops=Vec::new();
 for feature in 0..3 {let mut total=0.0;for _ in 0..5 {let column=test.clone().slice([0..100,feature..feature+1]).select(0,permutation(100,&d));let broken=test.clone().slice_assign([0..100,feature..feature+1],column);total+=baseline-accuracy(&m,broken,labels.clone());}drops.push(total/5.0);}
 println!("Тест {baseline:.3}, среднее падение по столбцам {:?}",drops);assert!(baseline>0.9);assert!(drops.iter().all(|v|v.is_finite()));assert!(drops[0]>drops[1] && drops[0]>drops[2]+0.15);
}
