use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device=device();
    let x=Tensor::<2>::random([240,2],Distribution::Normal(0.0,1.0),&device);
    let y=(x.clone().slice([0..240,0..1])+x.clone().slice([0..240,1..2])*0.5).greater_elem(0.0).int();
    let test=x.clone().slice([180..240]).reshape([60,1,2]);
    let train=x.slice([0..180]).reshape([1,180,2]);
    let distance=(test-train).powf_scalar(2.0).sum_dim(2).squeeze_dim::<2>(2);
    let (_,indices)=(-distance).topk_with_indices(3,1);
    let labels=y.clone().slice([0..180]).transpose().expand([60,180]).gather(1,indices.clone());
    let prediction=labels.float().mean_dim(1).greater_equal_elem(0.5).int();
    let accuracy=prediction.equal(y.slice([180..240])).float().mean().into_scalar::<f32>();
    println!("Верно на новых точках: {accuracy}; первые соседи: {:?}",indices.slice([0..1]).into_data());
    assert!(accuracy>0.8);
}
