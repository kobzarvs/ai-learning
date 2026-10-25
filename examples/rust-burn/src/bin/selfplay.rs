use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device = device();
    let payoff = Tensor::<2>::from_data([[0.0_f32,1.0,-1.0],[-1.0,0.0,1.0],[1.0,-1.0,0.0]],&device);
    let mut counts = [9.0_f32,3.0,1.0];
    let exploitability = |c: [f32;3]| {
        let total: f32 = c.iter().sum();

        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        payoff.clone().matmul(Tensor::<1>::from_data(c.map(|v|v/total),&device).reshape([3,1])).max().into_scalar::<f32>()
    };
    let before = exploitability(counts);
    for _ in 0..2000 {
        let total: f32 = counts.iter().sum();

        // Меняем форму тензора, сохраняя число и порядок элементов.
        let opponent = Tensor::<1>::from_data(counts.map(|v|v/total),&device).reshape([3,1]);
        let values = payoff.clone().matmul(opponent).into_data().try_to_vec::<f32>().unwrap();

        // Как argmax Python: при равенстве выбирается первый индекс.
        let mut action = 0; for a in 1..3 {if values[a]>values[action] {action=a;}}
        counts[action] += 1.0;
    }
    let total: f32 = counts.iter().sum(); let average = counts.map(|v|v/total);
    let after = exploitability(counts);
    println!("Средняя стратегия {average:?}; выгода лучшего ответа {before} -> {after}");
    assert_eq!(total,2013.0); assert!(after<0.05 && after<before);
    assert!(average.iter().all(|v|(v-1.0/3.0).abs()<0.05));
}
