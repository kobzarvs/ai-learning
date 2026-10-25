use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module,nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams},tensor::activation::tanh};
#[derive(Module,Debug)] struct Net {a:Linear,b:Linear}

// Задаём размеры линейного слоя; init создаёт его обучаемые веса.
impl Net {fn new(d:&Device)->Self{Self{a:LinearConfig::new(1,16).init(d),b:LinearConfig::new(16,1).init(d)}}fn forward(&self,x:Tensor<2>)->Tensor<2>{self.b.forward(tanh(self.a.forward(x)))}}

// Включаем запись вычислений, чтобы затем получить градиенты ошибки.
fn main(){let d=device().autodiff();let values:Vec<f32>=(0..80).map(|i|-1.0+2.0*i as f32/79.0).collect();let x=Tensor::<1>::from_floats(values.as_slice(),&d).reshape([80,1]);let y=x.clone().sin();let query=Tensor::<2>::from_floats([[-3.0],[0.0],[3.0]],&d);let mut answers=Vec::new();let mut losses=Vec::new();
 for seed in 0..5 {d.seed(seed);let mut m=Net::new(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.0).init();

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..200 {let loss=(m.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();let g=GradientsParams::from_grads(loss.backward(),&m);m=opt.step(0.02,m,g);}

 // Меняем форму тензора, сохраняя число и порядок элементов.
 let loss=(m.forward(x.clone())-y.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();losses.push(loss);answers.push(m.forward(query.clone()).reshape([3]).into_data().try_to_vec::<f32>().unwrap());}
 for j in 0..3 {let mean=answers.iter().map(|a|a[j]).sum::<f32>()/5.0;let std=(answers.iter().map(|a|(a[j]-mean).powi(2)).sum::<f32>()/4.0).sqrt();println!("x={}: среднее {mean:.4}, разброс {std:.4}",[-3,0,3][j]);assert!(mean.is_finite() && std.is_finite());}
 println!("Ответы независимых обучений {:?}; MSE {:?}",answers,losses);assert!(losses.iter().all(|l|l.is_finite() && *l<0.02));

 // Не требуем ненулевой дисперсии: одинаковые seed в упражнении законно дают
 // совпадение.
 println!("Согласие моделей не гарантирует правильность вне [-1,1]; разброс не является доверительным интервалом.");
}
