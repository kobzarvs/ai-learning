use burn::tensor::{Device, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams},tensor::activation::{log_softmax,softmax,log_sigmoid}};
#[derive(Module,Debug)]
struct Policy { logits:Param<Tensor<1>> }
fn main(){
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device=device().autodiff();

    // Регистрируем тензор как обучаемый параметр модели.
    let mut policy=Policy{logits:Param::from_tensor(Tensor::<1>::zeros([2],&device))};

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let reference=log_softmax(Tensor::<1>::zeros([2],&device),0).detach();

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let reference_before=reference.clone(); let mut optimizer=AdamConfig::new().init();

    // Берём часть тензора по указанным диапазонам индексов.
    let before=softmax(policy.logits.val(),0).slice([1..2]).into_scalar::<f32>();
    for _ in 0..100 {
        // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
        let logp=log_softmax(policy.logits.val(),0);
        let advantage=(logp.clone().slice([1..2])-logp.slice([0..1]))-(reference.clone().slice([1..2])-reference.clone().slice([0..1]));
        let loss=-log_sigmoid(advantage*0.5).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads=GradientsParams::from_grads(loss.backward(),&policy);policy=optimizer.step(0.05,policy,grads);
    }
    let probabilities=softmax(policy.logits.val(),0);

    // Читаем единственное значение тензора как обычное число Rust.
    let preferred=probabilities.clone().slice([1..2]).into_scalar::<f32>();
    println!("Вероятности {:?}; раньше ответ 1: {before}",probabilities.into_data().try_to_vec::<f32>().unwrap());
    assert!(preferred>0.95 && preferred>before);
    assert_eq!((reference-reference_before).abs().max().into_scalar::<f32>(),0.0);
}
