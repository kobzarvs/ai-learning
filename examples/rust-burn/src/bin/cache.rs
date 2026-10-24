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
    let mut scores=q.matmul(k.swap_dims(2,3))/(width as f64).sqrt();
    if causal {
        let mask: Vec<bool>=(0..length).flat_map(|i|(0..key_length).map(move|j|j>i)).collect();
        let mask=Tensor::<4,Bool>::from_data(TensorData::new(mask,[1,1,length,key_length]), &scores.device());
        scores=scores.mask_fill(mask.expand([batch,heads,length,key_length]), -1e9);
    }
    softmax(scores,3).matmul(v)
}

fn main() {
    let device = device();
    let x=Tensor::<4>::random([1,1,10,8],Distribution::Normal(0.0,1.0),&device);
    let q=x.clone()+0.1; let k=x.clone()-0.1; let v=x*0.5;
    let full=attention(q.clone(),k.clone(),v.clone(),true);
    let mut keys=Vec::new(); let mut values=Vec::new(); let mut outputs=Vec::new();
    for t in 0..10 {
        keys.push(k.clone().slice([0..1,0..1,t..t+1]));
        values.push(v.clone().slice([0..1,0..1,t..t+1]));
        outputs.push(attention(q.clone().slice([0..1,0..1,t..t+1]),Tensor::cat(keys.clone(),2),Tensor::cat(values.clone(),2),false));
    }
    let cached=Tensor::cat(outputs,2);
    let error=(full-cached).abs().max().into_scalar::<f32>();
    println!("Разница полного и пошагового расчёта: {error}"); assert!(error<1e-5);
}
