use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::LinearConfig,tensor::activation::softplus};
fn main(){
    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let device=device(); let target_layer=LinearConfig::new(2,4).init(&device); let speed_layer=LinearConfig::new(2,4).init(&device);
    let x=Tensor::<2>::ones([1,2],&device); let state=Tensor::<2>::zeros([1,4],&device);
    let target=target_layer.forward(x.clone()).tanh(); let speed=softplus(speed_layer.forward(x),1.0);
    let mut distances=Vec::new();
    for dt in [0.0,0.01,0.1,1.0] {
        let decay=(speed.clone()*(-dt)).exp();
        let next=decay.clone()*state.clone()+(decay.neg()+1.0)*target.clone();

        // Читаем единственное значение тензора как обычное число Rust.
        let distance=(next.clone()-target.clone()).powf_scalar(2.0).sum().into_scalar::<f32>(); distances.push(distance);
        if dt==0.0 { assert!((next.clone()-state.clone()).abs().max().into_scalar::<f32>()<1e-7); }
        println!("Интервал {dt}, состояние {:?}, расстояние² {distance:.6}",next.into_data());
    }
    assert!(distances[0]>1e-7); assert!(distances.windows(2).all(|d|d[1]<d[0]));
}
