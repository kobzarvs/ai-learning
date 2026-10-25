use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig}, optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::{silu}}};

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
    let device=device().autodiff(); let (x,y)=points(400,&device);
    let source=fit(Net::new(2,16,2,&device),x.clone().slice([0..300]),y.slice([0..300]));
    let encoder=source.hidden;

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let old_weight=encoder.weight.val().detach();
    let old_bias=encoder.bias.as_ref().unwrap().val().detach();

    // Отдельная голова — единственный Module, переданный optimizer.step.
    let features=silu(encoder.forward(x.clone())).detach();

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let labels=(x.clone().slice([0..400,0..1])-x.slice([0..400,1..2])*0.3).greater_elem(0.2).int().reshape([400]);
    let mut head=LinearConfig::new(16,2).init(&device);
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.001).init();
    let ce=CrossEntropyLossConfig::new().init(&device);
    for _ in 0..160 {
        let loss=ce.forward(head.forward(features.clone().slice([0..60])),labels.clone().slice([0..60]));

        // Обратный проход вычисляет градиенты ошибки.
        let grads=GradientsParams::from_grads(loss.backward(),&head); head=optimizer.step(0.02,head,grads);
    }

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let accuracy=head.forward(features.slice([300..400])).argmax(1).reshape([100]).equal(labels.slice([300..400])).float().mean().into_scalar::<f32>();

    // Читаем единственное значение тензора как обычное число Rust.
    let drift=(encoder.weight.val()-old_weight).abs().max().into_scalar::<f32>()+(encoder.bias.as_ref().unwrap().val()-old_bias).abs().max().into_scalar::<f32>();
    println!("Точность новой головы {accuracy:.3}; обучаемых чисел 34; изменение кодировщика {drift}");
    assert_eq!(drift,0.0); assert!(accuracy>0.85);
}
