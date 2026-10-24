use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let x = Tensor::<2>::ones([3, 8], &device);
    let base = Tensor::<2>::ones([8, 8], &device)*0.1;
    let a = Tensor::<2>::ones([8, 2], &device)*0.02;
    let b = Tensor::<2>::ones([2, 8], &device)*0.03;
    let separate = x.clone().matmul(base.clone()) + x.clone().matmul(a.clone()).matmul(b.clone());
    let merged = x.matmul(base+a.matmul(b));
    let error = (separate-merged).abs().max().into_scalar::<f32>();
    println!("Ошибка двух способов: {error}");
    assert!(error < 1e-5);
}
