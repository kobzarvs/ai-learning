use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device = device();
    let x = Tensor::<2>::from_data([[1.0_f32,2.,3.],[4.,5.,6.]], &device);
    let w = Tensor::<2>::from_data([[1.0_f32,0.],[0.,1.],[1.,1.]], &device);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let answer = x.matmul(w);
    println!("Форма: {:?}; ответ: {:?}", answer.dims(), answer.clone().into_data());
    assert_eq!(answer.into_data().try_to_vec::<f32>().unwrap(), vec![4.,5.,10.,11.]);
}
