use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device=device();
    let observed=Tensor::<1>::from_data([1.0_f32,1.,0.,1.,1.,0.,1.,1.],&device);

    // Читаем единственное значение тензора как обычное число Rust.
    let a=1+observed.sum().into_scalar::<f32>() as usize;
    let b=1+8-(a-1);

    // Для целых параметров гамма-распределение получаем как сумму -ln(U).
    // U — случайное число от 0 до 1.
    // X/(X+Y), где X и Y из гамма-распределений, даёт бета-распределение.
    let ga=-Tensor::<2>::random([10000,a],Distribution::Uniform(0.0000001,1.0),&device).log().sum_dim(1);
    let gb=-Tensor::<2>::random([10000,b],Distribution::Uniform(0.0000001,1.0),&device).log().sum_dim(1);
    let samples=(ga.clone()/(ga+gb)).squeeze::<1>().sort(0);

    // Берём часть тензора по указанным диапазонам индексов.
    let low=samples.clone().slice([249..250]).into_scalar::<f32>();
    let high=samples.slice([9749..9750]).into_scalar::<f32>();
    let mean=a as f32/(a+b) as f32;
    println!("Оценка вероятности: {mean}; 95% интервал: [{low}, {high}]");
    assert!((mean-0.7).abs()<1e-6 && low<0.7 && high>0.7 && low>0.3 && high<0.99);
}
