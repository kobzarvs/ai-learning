use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device = device();
    let decay = Tensor::<1>::from_data([0.1_f32,0.5,0.8,0.95], &device);
    let mut state = Tensor::<1>::zeros([4], &device);
    let mut outputs=Vec::new();
    for t in 0..30 {
        state=decay.clone()*state+if t==3 {1.0} else {0.0};
        outputs.push(state.clone().mean().into_scalar::<f32>());
    }
    println!("Ответ после импульса: {:?}", &outputs[3..12]);
    assert!(outputs[3]>outputs[8] && outputs[8]>outputs[20] && outputs[20]>0.0);
}
