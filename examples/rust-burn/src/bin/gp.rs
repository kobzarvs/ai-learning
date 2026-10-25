use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::linalg::svd;
fn kernel(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    ((a.unsqueeze_dim::<3>(1)-b.unsqueeze_dim::<3>(0)).powf_scalar(2.0)
        .sum_dim(2).squeeze_dim::<2>(2)*(-0.5)).exp()
}
fn main() {
    let device=device();
    let x=Tensor::<2>::from_data([[-1.0_f32],[0.0],[1.0]],&device); let y=x.clone().sin();
    let query=Tensor::<2>::from_data(burn::tensor::TensorData::new((0..21).map(|i| -3.0+i as f32*0.3).collect::<Vec<_>>(),[21,1]),&device);
    let k=kernel(x.clone(),x.clone())+Tensor::<2>::from_data([[0.01_f32,0.0,0.0],[0.0,0.01,0.0],[0.0,0.0,0.01]],&device);
    let cross=kernel(query,x);

    // SVD решает K z = rhs без замены обучения/ядра. Здесь K положительно определена.
    let (u,s,vt)=svd::<2,1>(k.clone(),100);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let inverse=(vt.transpose()*s.recip().unsqueeze_dim::<2>(0)).matmul(u.transpose());
    let alpha=inverse.clone().matmul(y.clone());

    // Читаем единственное значение тензора как обычное число Rust.
    let residual=(k.matmul(alpha.clone())-y).abs().max().into_scalar::<f32>();
    let mean=cross.clone().matmul(alpha);
    let variance=(cross.clone()*inverse.matmul(cross.transpose()).transpose()).sum_dim(1)*(-1.0)+1.0;
    let variance=variance.clamp_min(0.0).into_data().try_to_vec::<f32>().unwrap();
    let means=mean.into_data().try_to_vec::<f32>().unwrap();
    for i in [0,10,20] {println!("x={}, среднее {}, σ={}",-3.0+i as f32*0.3,means[i],variance[i].sqrt());}
    assert!(residual<1e-4 && variance[10]<0.02 && variance[0]>0.8 && variance[20]>0.8);
    assert!(means[10].abs()<1e-4 && (means[0]+means[20]).abs()<1e-3);
}
