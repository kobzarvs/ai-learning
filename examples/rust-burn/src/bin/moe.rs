use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::LinearConfig,tensor::activation::softmax};
fn main(){
    let device=device(); let x=Tensor::<2>::random([10,4],Distribution::Normal(0.0,1.0),&device);

    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let router=LinearConfig::new(4,3).init(&device);
    let experts:Vec<_>=(0..3).map(|_|LinearConfig::new(4,2).init(&device)).collect();

    // Преобразуем оценки в веса, сумма которых по выбранному измерению равна 1.
    let scores=softmax(router.forward(x.clone()),1);

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let choice=scores.clone().argmax(1).squeeze::<1>().into_data().try_into_vec_as::<i64>().unwrap();
    let mut counts=[0usize;3]; let mut rows=Vec::new(); let mut error=0.0f32;
    for (row,&expert) in choice.iter().enumerate(){
        let i=expert as usize; counts[i]+=1;

        // Берём часть тензора по указанным диапазонам индексов.
        let answer=experts[i].forward(x.clone().slice([row..row+1]));

        // Проверяем маршрутизацию против независимого вычисления полного батча
        // экспертом.
        error=error.max((answer.clone()-experts[i].forward(x.clone()).slice([row..row+1])).abs().max().into_scalar::<f32>());
        rows.push(answer);
    }
    let answer=Tensor::cat(rows,0);

    // Читаем единственное значение тензора как обычное число Rust.
    let probability_error=(scores.sum_dim(1)-1.0).abs().max().into_scalar::<f32>();
    println!("Выбор {choice:?}; нагрузка {counts:?}; размер {:?}",answer.dims());
    assert_eq!(counts.iter().sum::<usize>(),10); assert_eq!(answer.dims(),[10,2]);
    assert!(error<1e-5 && probability_error<1e-5);
}
