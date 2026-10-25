use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::tensor::Int;
fn metrics(y:Tensor<1,Int>,p:Tensor<1,Int>)->(f32,f32,f32){
 let yt=y.clone().into_data().try_to_vec::<i32>().unwrap();let pt=p.clone().into_data().try_to_vec::<i32>().unwrap();
 let (mut tp,mut fp,mut fn_)=(0usize,0usize,0usize);
 for (a,b) in yt.iter().zip(pt.iter()){if *b==1 {if *a==1 {tp+=1}else{fp+=1}}else if *a==1 {fn_+=1}}

 // Читаем единственное значение тензора как обычное число Rust.
 (p.equal(y).float().mean().into_scalar(),tp as f32/(tp+fp).max(1) as f32,tp as f32/(tp+fn_).max(1) as f32)
}
fn main(){let d=device();let labels:Vec<i32>=(0..100).map(|i|if i>=95 {1}else{0}).collect();let y=Tensor::<1,Int>::from_ints(labels.as_slice(),&d);
 let baseline=metrics(y.clone(),Tensor::<1,Int>::zeros([100],&d));
 let alarms:Vec<i32>=(0..100).map(|i|if i<6 || (95..99).contains(&i){1}else{0}).collect();let alternative=metrics(y,Tensor::from_ints(alarms.as_slice(),&d));
 println!("Без тревог accuracy/precision/recall {:?}; 10 тревог, 4 события {:?}",baseline,alternative);
 assert!((baseline.0-0.95).abs()<1e-6 && baseline.1==0.0 && baseline.2==0.0);
 assert!((alternative.0-0.93).abs()<1e-6 && (alternative.1-0.4).abs()<1e-6 && (alternative.2-0.8).abs()<1e-6);
}
