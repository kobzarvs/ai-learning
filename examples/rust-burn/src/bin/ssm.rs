use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let decay = Tensor::<1>::from_data([0.1_f32, 0.5, 0.8, 0.95], &device);
    let mut state = Tensor::<1>::zeros([4], &device);
    for step in 0..15 {
        let input = if step == 3 { 1.0 } else { 0.0 };
        state = decay.clone()*state + input;
        println!("{step}: {:.4}", state.clone().mean().into_scalar::<f32>());
    }
}
