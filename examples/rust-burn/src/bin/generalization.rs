use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
 // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
 fn new(d:&Device)->Self {Self{hidden:LinearConfig::new(2,64).init(d),output:LinearConfig::new(64,2).init(d)}}
 fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}
fn fit(mut m:Net,x:Tensor<2>,y:Tensor<1,Int>)->Net {
 // Ошибка классификации сравнивает оценки классов с верными метками.
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();let ce=CrossEntropyLossConfig::new().init(&x.device());

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..200 {let loss=ce.forward(m.forward(x.clone()),y.clone());let g=GradientsParams::from_grads(loss.backward(),&m);m=opt.step(0.02,m,g);}m
}

// Меняем форму тензора, сохраняя число и порядок элементов.
fn accuracy(m:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32 {let n=y.dims()[0];m.forward(x).argmax(1).reshape([n]).equal(y).float().mean().into_scalar()}
fn permutation(n:usize,d:&Device)->Tensor<1,Int>{
 let keys=Tensor::<1>::random([n],Distribution::Uniform(0.0,1.0),d).into_data().try_to_vec::<f32>().unwrap();
 let mut ix:Vec<usize>=(0..n).collect();ix.sort_by(|a,b|keys[*a].total_cmp(&keys[*b]));
 let ix:Vec<i32>=ix.into_iter().map(|i|i as i32).collect();Tensor::from_ints(ix.as_slice(),d)
}

// Включаем запись вычислений, чтобы затем получить градиенты ошибки.
fn main(){let d=device().autodiff();let x=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);let y=(x.clone().slice([0..400,0..1])+x.clone().slice([0..400,1..2])*0.5).greater_elem(0.0).int().reshape([400]);

 // Берём часть тензора по указанным диапазонам индексов.
 let tx=x.clone().slice([0..300]);let ty=y.clone().slice([0..300]);let shuffled=ty.clone().select(0,permutation(300,&d));

 // Читаем единственное значение тензора как обычное число Rust.
 assert_eq!(ty.clone().sum().into_scalar::<i32>(),shuffled.clone().sum().into_scalar::<i32>());
 let changed=300-ty.clone().equal(shuffled.clone()).int().sum().into_scalar::<i32>();
 println!("Изменено учебных меток: {changed}/300");assert!(changed>60);
 let true_model=fit(Net::new(&d),tx.clone(),ty.clone());let random_model=fit(Net::new(&d),tx.clone(),shuffled.clone());
 let true_train=accuracy(&true_model,tx.clone(),ty);let random_train=accuracy(&random_model,tx,shuffled);
 let true_test=accuracy(&true_model,x.clone().slice([300..400]),y.clone().slice([300..400]));let random_test=accuracy(&random_model,x.slice([300..400]),y.slice([300..400]));
 println!("Правильные метки train/test {true_train:.3}/{true_test:.3}; перемешанные {random_train:.3}/{random_test:.3}");assert!(true_train>0.9 && true_test>0.9);assert!(true_test>random_test+0.15);assert!(random_train.is_finite());
}
