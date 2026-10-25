use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::LinearConfig,optim::{AdamConfig,GradientsParams}};
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let state = Tensor::<2>::random([400,1],Distribution::Normal(0.0,1.0),&device);
    let action = Tensor::<2>::random([400,1],Distribution::Uniform(-1.0,1.0),&device);
    let next = state.clone()+action.clone()*0.3;
    let x = Tensor::cat(vec![state,action],1);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = LinearConfig::new(2,1).init(&device); let mut opt = AdamConfig::new().init();

    // Берём часть тензора по указанным диапазонам индексов.
    let error = |m: &burn::nn::Linear| (m.forward(x.clone().slice([320..400]))-next.clone().slice([320..400])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = error(&model);
    for _ in 0..300 {
        let loss = (model.forward(x.clone().slice([0..320]))-next.clone().slice([0..320])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.02,model,grads);
    }
    let after = error(&model);

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let candidates = Tensor::<1>::from_data((0..21).map(|i|-1.0+i as f32*0.1).collect::<Vec<_>>().as_slice(),&device).reshape([21,1]);
    let input = Tensor::cat(vec![Tensor::full([21,1],0.8,&device),candidates.clone()],1);

    // Читаем единственное значение тензора как обычное число Rust.
    let index = model.forward(input).powf_scalar(2.0).reshape([21]).argmin(0).into_scalar::<i64>() as usize;
    let chosen = candidates.into_data().try_to_vec::<f32>().unwrap()[index];
    println!("Ошибка прогноза: {before} -> {after}; действие: {chosen}");
    assert!(after<0.001 && after<before*0.01); assert!(chosen<=-0.9);
    assert!((0.8+0.3*chosen).abs()<0.8);
}
