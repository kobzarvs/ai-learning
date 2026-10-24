use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module, Param}, optim::{AdamConfig, GradientsParams}};
#[derive(Module, Debug)]
struct Adapter { a: Param<Tensor<2>>, b: Param<Tensor<2>> }
impl Adapter {
    fn forward(&self, x: Tensor<2>, base: Tensor<2>) -> Tensor<2> {
        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        x.clone().matmul(base) + x.matmul(self.a.val()).matmul(self.b.val())
    }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let x = Tensor::<2>::random([300, 8], Distribution::Normal(0.0, 1.0), &device);

    // Базовая матрица не входит в Adapter и не передаётся оптимизатору.
    let base = Tensor::<2>::random([8, 8], Distribution::Normal(0.0, 0.2), &device).detach();
    let base_before = base.clone();
    let true_update = Tensor::<2>::random([8, 2], Distribution::Normal(0.0, 1.0), &device)
        .matmul(Tensor::<2>::random([2, 8], Distribution::Normal(0.0, 1.0), &device))*0.1;

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let y = x.clone().matmul(base.clone()+true_update).detach();
    let mut adapter = Adapter {
        // Регистрируем тензор как обучаемый параметр модели.
        a: Param::from_tensor(Tensor::<2>::random([8, 2], Distribution::Normal(0.0, 0.01), &device)),
        b: Param::from_tensor(Tensor::<2>::zeros([2, 8], &device)) };
    let a_before = adapter.a.val().detach();

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut optimizer = AdamConfig::new().init();

    // Берём часть тензора по указанным диапазонам индексов.
    let test_x = x.clone().slice([240..300]);
    let test_y = y.clone().slice([240..300]);
    let before = (adapter.forward(test_x.clone(), base.clone())-test_y.clone())

        // Читаем единственное значение тензора как обычное число Rust.
        .powf_scalar(2.0).mean().into_scalar::<f32>();
    for _ in 0..250 {
        let prediction = adapter.forward(x.clone().slice([0..240]), base.clone());
        let loss = (prediction-y.clone().slice([0..240])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(), &adapter);

        // Обновляем параметры модели; первый аргумент задаёт скорость обучения.
        adapter = optimizer.step(0.04, adapter, grads);
    }
    let prediction = adapter.forward(test_x.clone(), base.clone());
    let after = (prediction.clone()-test_y).powf_scalar(2.0).mean().into_scalar::<f32>();
    let merged = base.clone()+adapter.a.val().matmul(adapter.b.val());
    let merge_error = (prediction-test_x.matmul(merged)).abs().max().into_scalar::<f32>();
    let base_change = (base-base_before).abs().max().into_scalar::<f32>();
    let a_change = (adapter.a.val()-a_before).abs().max().into_scalar::<f32>();
    let b_change = adapter.b.val().abs().max().into_scalar::<f32>();
    println!("MSE на 60 новых объектах: до {before:.6}, после {after:.6}");
    println!("Базовых весов: 64; обучаемых: {}; изменение базы: {base_change}", adapter.num_params());
    println!("Ошибка слияния адаптера: {merge_error}");
    assert!(after < before*0.02 && after < 0.001);
    assert_eq!(base_change, 0.0);
    assert!(a_change > 0.0001 && b_change > 0.0001);
    assert!(merge_error < 0.0001);
}
