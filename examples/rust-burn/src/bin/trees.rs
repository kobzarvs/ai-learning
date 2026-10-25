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
    let device = device();
    let samples = normal(480, &device);
    let x: Vec<[f32; 2]> = samples.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let y: Vec<bool> = x.iter().map(|p| p[0]+0.5*p[1]>0.0).collect();
    let mut best = (usize::MAX, 0, 0.0, false, false);
    for feature in 0..2 {
        for step in 0..30 {
            let threshold = -2.0+4.0*step as f32/29.0;
            let mut count = [0_usize; 2]; let mut positive = [0_usize; 2];
            for i in 0..180 {
                let side = usize::from(x[i][feature]>=threshold);
                count[side]+=1; positive[side]+=usize::from(y[i]);
            }
            if count.contains(&0) { continue; }
            let a = positive[0]*2>=count[0]; let b = positive[1]*2>=count[1];
            let error = (0..180).filter(|&i| (if x[i][feature]<threshold {a} else {b}) != y[i]).count();
            if error<best.0 { best=(error, feature, threshold, a, b); }
        }
    }
    let (error, feature, threshold, a, b)=best;
    let accuracy=(180..240).filter(|&i| (if x[i][feature]<threshold {a} else {b})==y[i]).count() as f32/60.0;
    println!("Признак {feature}, порог {threshold}, учебных ошибок {error}; новые строки: {accuracy}");
    assert!(error<50 && accuracy>0.7);
}
