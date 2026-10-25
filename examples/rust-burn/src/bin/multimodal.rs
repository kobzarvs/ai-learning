use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct Net{hidden:Linear,output:Linear}
impl Net{fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}}

// Меняем форму тензора, сохраняя число и порядок элементов.
fn accuracy(model:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32{model.forward(x).argmax(1).reshape([80]).equal(y).float().mean().into_scalar()}
fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();let image=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);let text=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);

 // Берём часть тензора по указанным диапазонам индексов.
 let y=(image.clone().slice([0..400,0..1])+text.clone().slice([0..400,0..1])).greater_elem(0.0).int().reshape([400]);let x=Tensor::cat(vec![image,text],1);

 // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
 let mut model=Net{hidden:LinearConfig::new(4,16).init(&d),output:LinearConfig::new(16,2).init(&d)};

 // Ошибка классификации сравнивает оценки классов с верными метками.
 let ce=CrossEntropyLossConfig::new().init(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();
 let before=accuracy(&model,x.clone().slice([320..400]),y.clone().slice([320..400]));

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..160{let loss=ce.forward(model.forward(x.clone().slice([0..320])),y.clone().slice([0..320]));let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);}
 let test=x.slice([320..400]);let labels=y.slice([320..400]);let both=accuracy(&model,test.clone(),labels.clone());
 let missing=Tensor::cat(vec![test.slice([0..80,0..2]),Tensor::<2>::zeros([80,2],&d)],1);let one=accuracy(&model,missing,labels);
 println!("До {before:.3}; оба источника {both:.3}; без текста {one:.3}");assert!(both>0.9 && both>one);
}
