use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::tensor::{TensorData, Bool, activation::softmax};
fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, causal: bool) -> Tensor<4> {
    let [batch, heads, length, width]=q.dims();
    let key_length=k.dims()[2];

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let mut scores=q.matmul(k.swap_dims(2,3))/(width as f64).sqrt();
    if causal {
        let mask: Vec<bool>=(0..length).flat_map(|i|(0..key_length).map(move|j|j>i)).collect();
        let mask=Tensor::<4,Bool>::from_data(TensorData::new(mask,[1,1,length,key_length]), &scores.device());
        scores=scores.mask_fill(mask.expand([batch,heads,length,key_length]), -1e9);
    }

    // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
    softmax(scores,3).matmul(v)
}

use burn::nn::LinearConfig;
fn main() {
    let device = device();
    let x=Tensor::<3>::random([2,6,16],Distribution::Normal(0.0,1.0),&device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let projection=LinearConfig::new(16,48).with_bias(false).init(&device).forward(x);

    // Берём часть тензора по указанным диапазонам индексов.
    let parts: Vec<Tensor<4>>=(0..3).map(|i|projection.clone().slice([0..2,0..6,i*16..(i+1)*16]).reshape([2,6,4,4]).swap_dims(1,2)).collect();
    let answer=attention(parts[0].clone(),parts[1].clone(),parts[2].clone(),true);
    let v=parts[2].clone();
    let changed=v.clone().slice_assign([0..2,0..4,5..6,0..4],v.slice([0..2,0..4,5..6,0..4])+100.0);
    let changed=attention(parts[0].clone(),parts[1].clone(),changed,true);

    // Читаем единственное значение тензора как обычное число Rust.
    let error=(answer.clone().slice([0..2,0..4,0..5])-changed.slice([0..2,0..4,0..5])).abs().max().into_scalar::<f32>();
    println!("Размер ответа: {:?}; изменение ранних ответов: {error}",answer.dims());
    assert_eq!(answer.dims(),[2,4,6,4]); assert!(error<1e-5);
}
