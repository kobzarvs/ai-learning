use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{TensorData, Int, activation::silu}};
#[derive(Module, Debug)]
struct Model { conv: Conv2d, output: Linear }
impl Model {
    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let n = x.dims()[0];

        // Меняем форму тензора, сохраняя число и порядок элементов.
        let features = silu(self.conv.forward(x)).mean_dim(2).mean_dim(3).reshape([n, 8]);
        self.output.forward(features)
    }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();

    // 240 картинок, два направления полос, пять возможных сдвигов.
    let labels: Vec<i32> = (0..240).map(|i| i % 2).collect();
    let mut pixels = vec![0.0_f32; 240*64];
    for i in 0..240 {
        let position = 1 + (i/2)%5;
        for row in 0..8 { for col in 0..8 {
            let axis = if labels[i]==0 { col } else { row };
            if axis >= position && axis < position+2 { pixels[i*64+row*8+col]=1.0; }
        }}
    }
    let x = Tensor::<4>::from_data(TensorData::new(pixels, [240, 1, 8, 8]), &device)
        + Tensor::<4>::random([240, 1, 8, 8], Distribution::Normal(0.0, 0.12), &device);
    let y = Tensor::<1, Int>::from_data(TensorData::new(labels, [240]), &device);
    let mut model = Model {
        // Задаём свёртку: число входных и выходных каналов и размер фильтра.
        conv: Conv2dConfig::new([1, 8], [3, 3]).with_padding(PaddingConfig2d::Same).init(&device),

        // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
        output: LinearConfig::new(8, 2).init(&device) };

    // Ошибка классификации сравнивает оценки классов с верными метками.
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..160 {
        // Берём часть тензора по указанным диапазонам индексов.
        let loss = loss_fn.forward(model.forward(x.clone().slice([0..180])), y.clone().slice([0..180]));

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model = optimizer.step(0.02, model, grads);
    }

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let accuracy = model.forward(x.slice([180..240])).argmax(1).squeeze::<1>()

        // Читаем единственное значение тензора как обычное число Rust.
        .equal(y.slice([180..240])).float().mean().into_scalar::<f32>();
    println!("Верно на 60 новых картинках: {accuracy}; обучаемых чисел: {}", model.num_params());
    assert!(accuracy > 0.9, "CNN не научилась различать полосы");
}
