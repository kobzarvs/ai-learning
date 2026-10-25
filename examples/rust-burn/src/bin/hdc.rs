use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main() {
    let device=device();
    let x=Tensor::<2>::random([300,2],Distribution::Normal(0.0,1.0),&device);
    let points=x.clone().into_data().try_to_vec::<f32>().unwrap();
    let labels:Vec<usize>=points.chunks_exact(2).map(|p|usize::from(p[0]+0.5*p[1]>0.0)).collect();
    let projection=Tensor::<2>::random([2,1024],Distribution::Normal(0.0,1.0),&device);

    // Матрично умножаем тензоры: внутренние размеры должны совпадать.
    let codes=x.matmul(projection).greater_equal_elem(0.0).float()*2.0-1.0;
    let raw=codes.clone().into_data().try_to_vec::<f32>().unwrap();
    let mut proto=vec![0.0_f32;2048]; let mut counts=[0;2];
    for i in 0..240 {counts[labels[i]]+=1; for j in 0..1024 {proto[labels[i]*1024+j]+=raw[i*1024+j];}}
    assert!(counts.iter().all(|n|*n>50));

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let prototypes=Tensor::<1>::from_data(proto.as_slice(),&device).reshape([2,1024]);
    let norm=prototypes.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-8);
    let prototypes=prototypes/norm;

    // Берём часть тензора по указанным диапазонам индексов.
    let scores=(codes.slice([240..300])/32.0).matmul(prototypes.transpose());

    // Выбираем индекс наибольшей оценки — предсказанную категорию.
    let predicted=scores.argmax(1).into_data().try_to_vec::<i32>().unwrap();
    let accuracy=predicted.iter().zip(&labels[240..]).filter(|(a,b)|**a as usize==**b).count() as f32/60.0;
    println!("Точность новых точек: {accuracy}"); assert!(accuracy>0.85);
}
