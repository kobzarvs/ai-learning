use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn simulate(current:f32)->Vec<i32> {
    let device = device(); let mut voltage = Tensor::<1>::zeros([1],&device); let mut spikes=Vec::new();
    for _ in 0..20 {
        voltage=voltage*0.9+current;
        let spike=voltage.clone().greater_equal_elem(1.0).float();

        // Читаем единственное значение тензора как обычное число Rust.
        voltage=voltage-spike.clone(); spikes.push(spike.into_scalar::<f32>() as i32);
        let v=voltage.clone().into_scalar::<f32>(); assert!(v>=0.0 && v<1.0);
    }
    spikes
}
fn main() {
    let spikes=simulate(0.4); let count:i32=spikes.iter().sum();
    println!("Импульсы: {spikes:?}; всего: {count}");
    assert_eq!(spikes,vec![0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]);
    assert!(simulate(0.2).iter().sum::<i32>()<count && simulate(0.8).iter().sum::<i32>()>count);
}
