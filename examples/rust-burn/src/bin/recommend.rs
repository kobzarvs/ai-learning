use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param},optim::{AdamWConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Factors{users:Param<Tensor<2>>,items:Param<Tensor<2>>}

// Матрично умножаем тензоры: внутренние размеры должны совпадать.
impl Factors{fn forward(&self)->Tensor<2>{self.users.val().matmul(self.items.val().transpose())}}
fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();

 // Отделяем тензор от графа градиентов, сохраняя его значения.
 let rating=Tensor::<2>::random([12,3],Distribution::Normal(0.0,1.0),&d).matmul(Tensor::<2>::random([10,3],Distribution::Normal(0.0,1.0),&d).transpose()).detach();
 let mask=Tensor::<2>::random([12,10],Distribution::Uniform(0.0,1.0),&d).lower_elem(0.8).float();let hidden=mask.clone().neg()+1.0;

 // Читаем единственное значение тензора как обычное число Rust.
 let known_count=mask.clone().sum();let hidden_count=hidden.clone().sum();assert!(hidden_count.clone().into_scalar::<f32>()>0.0);

 // Регистрируем тензор как обучаемый параметр модели.
 let mut model=Factors{users:Param::from_tensor(Tensor::<2>::random([12,3],Distribution::Normal(0.0,1.0),&d)),items:Param::from_tensor(Tensor::<2>::random([10,3],Distribution::Normal(0.0,1.0),&d))};
 let before=((model.forward()-rating.clone()).powf_scalar(2.0)*mask.clone()).sum()/known_count.clone();
 let baseline=(rating.clone().powf_scalar(2.0)*hidden.clone()).sum()/hidden_count.clone();
 let mut opt=AdamWConfig::new().with_weight_decay(0.01).init();

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..400{let loss=((model.forward()-rating.clone()).powf_scalar(2.0)*mask.clone()).sum()/known_count.clone();let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.04,model,grads);}
 let prediction=model.forward();let train=((prediction.clone()-rating.clone()).powf_scalar(2.0)*mask.clone()).sum()/known_count;
 let test=((prediction.clone()-rating).powf_scalar(2.0)*hidden.clone()).sum()/hidden_count;
 let a=before.into_scalar::<f32>();let b=train.into_scalar::<f32>();let c=test.into_scalar::<f32>();let z=baseline.into_scalar::<f32>();

 // Ранжируем только предметы, оценки которых не показывали при обучении.
 let scores=prediction.into_data().try_to_vec::<f32>().unwrap();let held=hidden.into_data().try_to_vec::<f32>().unwrap();
 for user in 0..12{let mut candidates:Vec<_>=(0..10).filter(|item|held[user*10+item]>0.5).collect();candidates.sort_by(|a,b|scores[user*10+b].total_cmp(&scores[user*10+a]));assert!(candidates.windows(2).all(|p|scores[user*10+p[0]]>=scores[user*10+p[1]]));println!("Пользователь {user}: скрытые кандидаты {candidates:?}");}
 println!("MSE известных до {a:.4}, после {b:.4}; скрытых {c:.4}; нулевой прогноз {z:.4}");assert!(b<a*0.1 && c.is_finite());
}
