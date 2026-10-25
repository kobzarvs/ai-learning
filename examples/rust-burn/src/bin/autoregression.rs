use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig}, optim::{AdamWConfig, GradientsParams}, tensor::Int};
#[derive(Module, Debug)]
struct Language { embedding: Embedding, output: Linear }
impl Language {
    fn forward(&self, x: Tensor<2,Int>) -> Tensor<2> { self.output.forward(self.embedding.forward(x).squeeze_dim::<2>(1)) }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let tokens: Vec<i32> = "abcabcabcabcabcabc".bytes().map(|c|(c-b'a') as i32).collect();

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let x = Tensor::<1,Int>::from_data(tokens[..17].as_ref(),&device).reshape([17,1]);
    let y = Tensor::<1,Int>::from_data(tokens[1..].as_ref(),&device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = Language { embedding: EmbeddingConfig::new(3,8).init(&device), output: LinearConfig::new(8,3).init(&device) };
    let mut opt = AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.001).init();

    // Ошибка классификации сравнивает оценки классов с верными метками.
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    for _ in 0..160 {
        let loss = loss_fn.forward(model.forward(x.clone()),y.clone());

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(0.02,model,grads);
    }

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let accuracy = model.forward(x).argmax(1).squeeze::<1>().equal(y).float().mean().into_scalar::<f32>();
    let mut current = 0_i32;
    let mut generated = String::from("a");
    for _ in 0..12 {
        // Читаем единственное значение тензора как обычное число Rust.
        current = model.forward(Tensor::<2,Int>::from_data([[current]],&device)).argmax(1).into_scalar::<i32>();
        generated.push((b'a'+current as u8) as char);
    }
    println!("Продолжение: {generated}; точность: {accuracy}");
    assert!(accuracy>0.99);
    assert_eq!(generated,"abcabcabcabca");
}
