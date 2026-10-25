use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::Module, nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams}, tensor::{Int, TensorData, activation::silu}};
#[derive(Module, Debug)]
struct Agent { hidden: Linear, output: Linear }
impl Agent {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.output.forward(silu(self.hidden.forward(x)))
    }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let features = Tensor::<2>::eye(5, &device);

    // Берём часть тензора по указанным диапазонам индексов.
    let states = features.clone().slice([0..4, 0..5]);
    let next_ids = Tensor::<1, Int>::from_data([0_i64, 1, 0, 2, 1, 3, 2, 4], &device);
    let rewards = Tensor::<2>::from_data(TensorData::new(
        vec![-0.02_f32, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, 1.0], [4, 2]), &device);
    let alive = Tensor::<2>::from_data(TensorData::new(
        vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [4, 2]), &device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let mut model = Agent { hidden: LinearConfig::new(5, 16).init(&device),
                            output: LinearConfig::new(16, 2).init(&device) };

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..300 {
        // Цель не участвует в обратном распространении градиентов.
        let future = model.forward(features.clone()).detach().max_dim(1)

            // Меняем форму тензора, сохраняя число и порядок элементов.
            .squeeze::<1>().select(0, next_ids.clone()).reshape([4, 2]);
        let target = rewards.clone() + future * alive.clone() * 0.95;
        let loss = (model.forward(states.clone()) - target).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &model);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        model = optimizer.step(0.02, model, grads);
    }

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let actions = model.forward(states).argmax(1).squeeze::<1>()
        .into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
    println!("Действия в клетках 0–3: {actions:?}");
    let mut state = 0_usize;
    let mut route = vec![state];
    for _ in 0..20 {
        if state == 4 { break; }
        state = if actions[state] == 1 { (state + 1).min(4) } else { state.saturating_sub(1) };
        route.push(state);
    }
    println!("Маршрут: {route:?}; выход достигнут: {}", state == 4);
    assert_eq!(actions, vec![1, 1, 1, 1], "Агент должен идти вправо к награде");
}
