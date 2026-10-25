use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn score(x:[f32;2])->f32 {(x[0]-1.5).powi(2)+(x[1]+0.7).powi(2)}
fn main() {
    let device = device();
    let raw = Tensor::<2>::random([100,2],Distribution::Normal(0.0,3.0),&device).into_data().try_to_vec::<f32>().unwrap();
    let mut population:Vec<[f32;2]> = raw.chunks_exact(2).map(|p|[p[0],p[1]]).collect();
    let initial = population.iter().map(|p|score(*p)).fold(f32::INFINITY,f32::min);
    let mut previous = initial;
    for _ in 0..50 {
        population.sort_by(|a,b|score(*a).total_cmp(&score(*b)));
        let elite = population[..10].to_vec();
        let noise = Tensor::<2>::random([90,2],Distribution::Normal(0.0,0.08),&device).into_data().try_to_vec::<f32>().unwrap();
        population = elite.clone();
        for i in 0..90 {population.push([elite[i%10][0]+noise[2*i],elite[i%10][1]+noise[2*i+1]]);}
        let current = population.iter().map(|p|score(*p)).fold(f32::INFINITY,f32::min);
        assert!(current<=previous); previous=current;
    }
    population.sort_by(|a,b|score(*a).total_cmp(&score(*b)));
    println!("Лучший: {:?}; ошибка {initial} -> {}",population[0],score(population[0]));
    assert!(score(population[0])<0.001);
}
