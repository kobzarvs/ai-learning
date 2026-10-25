use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::tensor::activation::softmax;
fn main() {
    let device = device();
    let logits = Tensor::<2>::from_data([[1.0_f32,2.0,5.0]],&device);
    let forbidden = Tensor::<2,burn::tensor::Bool>::from_data([[false,false,true]],&device);
    let restricted = logits.mask_fill(forbidden,f32::NEG_INFINITY);

    // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
    let probabilities = softmax(restricted,1);

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let action = probabilities.clone().argmax(1).into_scalar::<i32>();
    let p = probabilities.into_data().try_to_vec::<f32>().unwrap();
    println!("Вероятности: {p:?}; выбрано: {}",["читать","отправить","удалить"][action as usize]);
    assert_eq!(action,1); assert_eq!(p[2],0.0); assert!((p.iter().sum::<f32>()-1.0).abs()<1e-6);
}
