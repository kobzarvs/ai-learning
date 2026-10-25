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

fn uniform(n: usize, device: &Device) -> Vec<f32> {
    Tensor::<1>::random([n], Distribution::Uniform(0.0, 1.0), device)
        .into_data().try_to_vec::<f32>().unwrap()
}

fn depth(x: &[[f32;2]], target: usize, draws: &[f32], cursor: &mut usize) -> usize {
    let mut active: Vec<usize>=(0..x.len()).collect();
    for level in 0..30 {
        if active.len()<=1 {return level;}
        let feature=usize::from(draws[*cursor]>=0.5); *cursor+=1;
        let low=active.iter().map(|&i| x[i][feature]).fold(f32::INFINITY,f32::min);
        let high=active.iter().map(|&i| x[i][feature]).fold(f32::NEG_INFINITY,f32::max);
        if high==low {return level;}
        let cut=low+draws[*cursor]*(high-low); *cursor+=1;
        let side=x[target][feature]<cut;
        active.retain(|&i| (x[i][feature]<cut)==side);
        assert!(active.contains(&target));
    } 30
}
fn main() {
    let device=device(); let noise=normal(200,&device);
    let mut x: Vec<[f32;2]>=noise.chunks_exact(2).map(|p| [p[0]*0.3,p[1]*0.3]).collect(); x.push([5.0,5.0]);
    let draws=uniform(12000,&device); let mut cursor=0;
    let ordinary=(0..100).map(|_| depth(&x,0,&draws,&mut cursor) as f32).sum::<f32>()/100.0;
    let outlier=(0..100).map(|_| depth(&x,100,&draws,&mut cursor) as f32).sum::<f32>()/100.0;
    println!("Средняя глубина: обычная {ordinary}; далёкая {outlier}");
    assert!(outlier<ordinary && outlier<4.0 && ordinary>4.0);
}
