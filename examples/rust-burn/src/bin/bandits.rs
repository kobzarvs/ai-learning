use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device = device();

    // Gamma(k,1) при целом k — сумма k независимых Exp(1).
    // Все posterior shapes целые: исходные единицы плюс Bernoulli-наблюдения.
    let draws = Tensor::<1>::random([1_000_000],Distribution::Uniform(0.0000001,1.0),&device).into_data().try_to_vec::<f32>().unwrap();
    let mut cursor = 0;
    let mut uniform = || {let u=draws[cursor]; cursor+=1; u};
    let chance = [0.2_f32,0.5,0.8]; let mut success = [1_usize;3]; let mut failure = [1_usize;3]; let mut count = [0_usize;3];
    for _ in 0..800 {
        let mut guesses = [0.0_f32;3];
        for a in 0..3 {
            let ga: f32 = (0..success[a]).map(|_|-uniform().ln()).sum();
            let gb: f32 = (0..failure[a]).map(|_|-uniform().ln()).sum();
            guesses[a] = ga/(ga+gb);
        }
        let action = (0..3).max_by(|&a,&b|guesses[a].total_cmp(&guesses[b])).unwrap();
        let reward = uniform()<chance[action];
        if reward {success[action]+=1;} else {failure[action]+=1;} count[action]+=1;
    }
    let means: Vec<f32> = (0..3).map(|a|success[a] as f32/(success[a]+failure[a]) as f32).collect();
    println!("Выборы: {count:?}; posterior means: {means:?}");
    assert_eq!(count.iter().sum::<usize>(),800);
    for a in 0..3 {assert_eq!(success[a]+failure[a],count[a]+2);}
    assert!(count[2]>count[0] && count[2]>count[1] && count[2]>500);
    assert!((means[2]-0.8).abs()<0.08);
}
