use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn summary(flag: Tensor<1>, time: Tensor<1>) -> (f32,f32) {
    let n=flag.dims()[0] as f64;
    let treated=flag.clone().sum(); let control=flag.clone()*(-1.0)+1.0;
    let difference=((flag.clone()*time.clone()).sum()/treated.clone()

        // Читаем единственное значение тензора как обычное число Rust.
        -(control.clone()*time.clone()).sum()/control.sum()).into_scalar::<f32>();
    let centered_flag=flag.clone()-flag.mean(); let centered_time=time.clone()-time.mean();
    let covariance=(centered_flag.clone()*centered_time.clone()).sum()/n;
    let scale=((centered_flag.powf_scalar(2.0).sum()/n)*(centered_time.powf_scalar(2.0).sum()/n)).sqrt();
    ((covariance/scale).into_scalar::<f32>(),difference)
}
fn main() {
    let device=device(); let n=1000;
    let difficulty=Tensor::<1>::random([n],Distribution::Uniform(1.0,5.0),&device);
    let assignment=Tensor::<1>::random([n],Distribution::Uniform(0.0,1.0),&device);
    let noise=Tensor::<1>::random([n],Distribution::Uniform(-1.0,1.0),&device);
    let minutes_saved=4.0;
    let completion_time=|instruction: Tensor<1>| difficulty.clone()*10.0+20.0-instruction*minutes_saved+noise.clone();
    let changes=completion_time(Tensor::<1>::ones([n],&device))-completion_time(Tensor::<1>::zeros([n],&device));
    let intervention=changes.clone().mean().into_scalar::<f32>();
    assert!((changes+minutes_saved).abs().max().into_scalar::<f32>()<1e-5);
    let mut differences=Vec::new();
    for random_assignment in [false,true] {
        let probability=if random_assignment {Tensor::<1>::ones([n],&device)*0.5} else {(difficulty.clone()-1.0)*0.2+0.1};
        let flag=assignment.clone().lower(probability).float();
        let (correlation,difference)=summary(flag.clone(),completion_time(flag)); differences.push(difference);
        println!("Случайное назначение {random_assignment}; корреляция {correlation}; разница групп {difference}; парное вмешательство {intervention}");
    }
    assert!((intervention+minutes_saved as f32).abs()<1e-5);
    assert!(differences[0]>0.0 && (differences[1]+minutes_saved as f32).abs()<2.5);
}
