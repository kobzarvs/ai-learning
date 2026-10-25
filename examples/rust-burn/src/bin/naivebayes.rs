use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn normal(n: usize, device: &Device) -> Vec<f32> {
    Tensor::<1>::random([n], Distribution::Normal(0.0, 1.0), device)
        .into_data().try_to_vec::<f32>().unwrap()
}

fn main() {
    let device=device(); let noise=normal(480,&device);

    // Чередование классов даёт перемешанную выборку с обоими классами в каждом срезе.
    let x: Vec<[f32;2]>=(0..240).map(|i| {let shift=if i%2==0 {-1.0} else {1.0}; [noise[2*i]+shift,noise[2*i+1]+shift]}).collect();
    let mut means=[[0.0_f32;2];2]; let mut variance=[[0.0_f32;2];2]; let mut counts=[0_usize;2];
    for i in 0..180 { let k=i%2; counts[k]+=1; for j in 0..2 {means[k][j]+=x[i][j];} }
    for k in 0..2 {for j in 0..2 {means[k][j]/=counts[k] as f32;}}
    for i in 0..180 {for j in 0..2 {variance[i%2][j]+=(x[i][j]-means[i%2][j]).powi(2);}}
    for k in 0..2 {for j in 0..2 {variance[k][j]=variance[k][j]/(counts[k]-1) as f32+1e-4;}}

    // Burn вычисляет логарифмы плотностей одновременно для 60 × 2 × 2 значений.
    let test=Tensor::<2>::from_data(burn::tensor::TensorData::new(x[180..].iter().flatten().copied().collect::<Vec<_>>(),[60,2]),&device).unsqueeze_dim::<3>(1);
    let mu=Tensor::<2>::from_data(means,&device).unsqueeze_dim::<3>(0);
    let var=Tensor::<2>::from_data(variance,&device).unsqueeze_dim::<3>(0);
    let prior=Tensor::<1>::from_data([counts[0] as f32/180.0,counts[1] as f32/180.0],&device).log().unsqueeze_dim::<2>(0);
    let score=(((test-mu).powf_scalar(2.0)/var.clone()+var.log()).sum_dim(2).squeeze_dim::<2>(2)*(-0.5))+prior;

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let predicted=score.argmax(1).into_data().try_to_vec::<i32>().unwrap();
    let accuracy=predicted.iter().enumerate().filter(|&(i,k)| *k as usize==(180+i)%2).count() as f32/60.0;
    println!("Средние: {means:?}; точность: {accuracy}"); assert!(accuracy>0.8);
}
