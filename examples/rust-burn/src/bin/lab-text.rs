use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::{LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,TensorData}};
fn encode(texts:&[&str],vocab:&[&str],d:&Device)->Tensor<2>{
 let mut values=vec![0.0f32;texts.len()*vocab.len()];
 for (i,text) in texts.iter().enumerate(){for word in text.split_whitespace(){if let Some(j)=vocab.iter().position(|w|*w==word){values[i*vocab.len()+j]+=1.0;}}}
 Tensor::from_data(TensorData::new(values,[texts.len(),vocab.len()]),d)
}
fn main(){
 // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
 let d=device().autodiff();
 let texts=["включи свет","свет включить","зажги лампу","лампу включить","выключи свет","свет выключить","погаси лампу","лампу выключить"];
 let mut vocab:Vec<_>=texts.iter().flat_map(|s|s.split_whitespace()).collect();vocab.sort();vocab.dedup();
 let x=encode(&texts,&vocab,&d);let y=Tensor::<1,Int>::from_data([1,1,1,1,0,0,0,0],&d);

 // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
 let mut model=LinearConfig::new(vocab.len(),2).init(&d);let ce=CrossEntropyLossConfig::new().init(&d);

 // Читаем единственное значение тензора как обычное число Rust.
 let before=ce.forward(model.forward(x.clone()),y.clone()).into_scalar::<f32>();
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();

 // Связываем градиенты вычислительного графа с параметрами этой модели.
 for _ in 0..160{let loss=ce.forward(model.forward(x.clone()),y.clone());let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);}
 let after=ce.forward(model.forward(x),y).into_scalar::<f32>();
 let checks=["включи лампу","выключи лампу","пожалуйста свет включить"];

 // Меняем форму тензора, сохраняя число и порядок элементов.
 let prediction=model.forward(encode(&checks,&vocab,&d)).argmax(1).reshape([3]).into_data().try_into_vec_as::<i64>().unwrap();
 println!("Ошибка до {before:.4}, после {after:.4}; команды {prediction:?}");
 assert!(after<before*0.2);assert_eq!(prediction,vec![1,0,1]);

 // Неизвестное слово игнорируется; это ограничение представления, а не понимание
 // отрицания.
 assert_eq!(encode(&["телепортируй"],&vocab,&d).sum().into_scalar::<f32>(),0.0);
 assert_eq!((encode(&["не включи свет"],&vocab,&d)-encode(&["включи свет"],&vocab,&d)).abs().max().into_scalar::<f32>(),0.0);
}
