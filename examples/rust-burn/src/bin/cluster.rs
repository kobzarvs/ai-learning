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
    let device=device(); let noise=normal(400,&device);
    let x: Vec<[f32;2]>=(0..200).map(|i| {let shift=if i<100 {-1.0} else {1.0}; [noise[2*i]*0.3+shift,noise[2*i+1]*0.3+shift]}).collect();
    let mut centers=[x[7],x[137]]; let mut labels=vec![0;200]; let mut first=0.0;
    let data=Tensor::<2>::from_data(burn::tensor::TensorData::new(x.iter().flatten().copied().collect::<Vec<_>>(),[200,2]),&device);
    for iteration in 0..20 {
        let c=Tensor::<2>::from_data(centers,&device);
        let distance=(data.clone().unsqueeze_dim::<3>(1)-c.unsqueeze_dim::<3>(0)).powf_scalar(2.0).sum_dim(2).squeeze_dim::<2>(2);

        // Читаем единственное значение тензора как обычное число Rust.
        if iteration==0 { first=distance.clone().min_dim(1).mean().into_scalar::<f32>(); }
        labels=distance.argmin(1).into_data().try_to_vec::<i32>().unwrap().iter().map(|&k| k as usize).collect();
        let mut sum=[[0.0;2];2]; let mut count=[0;2];
        for (p,&k) in x.iter().zip(&labels) {count[k]+=1; for j in 0..2 {sum[k][j]+=p[j];}}
        for k in 0..2 {if count[k]>0 {for j in 0..2 {centers[k][j]=sum[k][j]/count[k] as f32;}}}
    }
    let error=x.iter().zip(&labels).map(|(p,&k)| (p[0]-centers[k][0]).powi(2)+(p[1]-centers[k][1]).powi(2)).sum::<f32>()/200.0;
    let count=labels.iter().filter(|&&k| k==0).count();
    println!("Центры {centers:?}; размеры [{count}, {}]; ошибка {first} → {error}",200-count);
    assert!(error<=first+1e-5 && error<0.3 && (90..=110).contains(&count));
    assert!(centers.iter().all(|c| (c[0].abs()-1.0).abs()<0.15 && (c[1].abs()-1.0).abs()<0.15));
    assert!(centers[0][0]*centers[1][0]<0.0);
}
