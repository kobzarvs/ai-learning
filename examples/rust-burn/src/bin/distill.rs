use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::{silu,softmax,log_softmax}}};

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

// Выбираем индекс наибольшей оценки — предсказанную категорию.
fn acc_linear(model:&Linear,x:Tensor<2>,y:Tensor<1,Int>)->f32 {model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()}

fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();let (x,y)=points(400,&d);let teacher=fit_net(Net::new(2,32,&d),x.clone().slice([0..300]),y.clone().slice([0..300]),160);

 // Отделяем тензор от графа градиентов, сохраняя его значения.
 let teacher_w=teacher.hidden.weight.val().detach();let mut student=LinearConfig::new(2,2).init(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.01).init();let ce=CrossEntropyLossConfig::new().init(&d);let temperature=2.0;

 // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
 let soft=softmax(teacher.forward(x.clone().slice([0..300]))/temperature,1).detach();

 // Обратный проход вычисляет градиенты ошибки.
 for _ in 0..180 {let logits=student.forward(x.clone().slice([0..300]));let logp=log_softmax(logits.clone()/temperature,1);let kl=(soft.clone()*(soft.clone().clamp_min(1e-8).log()-logp)).sum()/300.0*(temperature*temperature);let loss=kl*0.7+ce.forward(logits,y.clone().slice([0..300]))*0.3;let grads=GradientsParams::from_grads(loss.backward(),&student);student=opt.step(0.03,student,grads);}

 // Читаем единственное значение тензора как обычное число Rust.
 let t=acc_net(&teacher,x.clone().slice([300..400]),y.clone().slice([300..400]));let s=acc_linear(&student,x.slice([300..400]),y.slice([300..400]));let teacher_params=teacher.num_params();let student_params=student.num_params();println!("Учитель {t:.3}, {teacher_params} параметров; ученик {s:.3}, {student_params} параметров");assert!(s>0.9 && t>0.9 && student_params<teacher_params);assert!((teacher.hidden.weight.val()-teacher_w).abs().max().into_scalar::<f32>()<1e-7);
}
