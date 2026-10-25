use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig}, optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::{silu,softmax}}};

#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
    fn new(inputs:usize,width:usize,outputs:usize,device:&Device)->Self {
        // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
        Self{hidden:LinearConfig::new(inputs,width).init(device),output:LinearConfig::new(width,outputs).init(device)}
    }
    fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}

fn points(n:usize,device:&Device)->(Tensor<2>,Tensor<1,Int>){
    let x=Tensor::<2>::random([n,2],Distribution::Normal(0.0,1.0),device);

    // Берём часть тензора по указанным диапазонам индексов.
    let y=(x.clone().slice([0..n,0..1])+x.clone().slice([0..n,1..2])*0.5).greater_elem(0.0).int().reshape([n]);
    (x,y)
}

fn accuracy(model:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32 {
    // Меняем форму тензора, сохраняя число и порядок элементов.
    model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()
}

fn fit(mut model:Net,x:Tensor<2>,y:Tensor<1,Int>)->Net {
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.001).init();

    // Ошибка классификации сравнивает оценки классов с верными метками.
    let ce=CrossEntropyLossConfig::new().init(&x.device());
    for _ in 0..160 {
        let loss=ce.forward(model.forward(x.clone()),y.clone());

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model=optimizer.step(0.02,model,grads);
    }
    model
}

fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff(); let (x,y)=points(500,&device);
    let model=fit(Net::new(2,16,2,&device),x.clone().slice([0..30]),y.clone().slice([0..30]));
    let before=accuracy(&model,x.clone().slice([400..500]),y.clone().slice([400..500]));

    // Один и тот же экземпляр сначала служит учителем, затем дообучается, как в Python.
    let probabilities=softmax(model.forward(x.clone().slice([30..400])).detach(),1);
    let (confidence,pseudo)=probabilities.max_dim_with_indices(1);
    let values=confidence.reshape([370]).into_data().try_to_vec::<f32>().unwrap();
    let chosen:Vec<i32>=values.iter().enumerate().filter(|(_,v)|**v>0.95).map(|(i,_)|i as i32).collect();
    assert!(!chosen.is_empty(),"Уверенные псевдометки должны существовать");
    let indices=Tensor::<1,Int>::from_ints(chosen.as_slice(),&device);
    let ax=Tensor::cat(vec![x.clone().slice([0..30]),x.clone().slice([30..400]).select(0,indices.clone())],0);
    let ay=Tensor::cat(vec![y.clone().slice([0..30]),pseudo.reshape([370]).select(0,indices)],0);
    let model=fit(model,ax,ay);
    let after=accuracy(&model,x.slice([400..500]),y.slice([400..500]));
    println!("Псевдометок {}; точность до {before:.3}, после {after:.3}",chosen.len());
    assert!(before>0.8 && after>0.8,"Псевдометки не гарантируют улучшения");
}
