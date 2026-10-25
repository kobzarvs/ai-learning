use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{nn::LinearConfig,module::Param};
fn main(){
    // Задаём размеры линейного слоя; init создаёт его обучаемые веса.
    let device=device(); let mut model=LinearConfig::new(20,10).init(&device);
    let x=Tensor::<2>::random([30,20],Distribution::Normal(0.0,1.0),&device);
    let before=model.forward(x.clone()); let weights=model.weight.val();
    let mut sorted=weights.clone().abs().into_data().try_to_vec::<f32>().unwrap();
    sorted.sort_by(|a,b|a.total_cmp(b));

    // torch.quantile(..., 0.5): линейная интерполяция двух центральных значений.
    let threshold=(sorted[99]+sorted[100])*0.5;

    // Регистрируем тензор как обучаемый параметр модели.
    model.weight=Param::from_tensor(weights.clone()*weights.abs().greater_equal_elem(threshold).float());
    let after=model.forward(x);

    // Читаем единственное значение тензора как обычное число Rust.
    let zeros=model.weight.val().equal_elem(0.0).int().sum().into_scalar::<i32>();
    let bytes=model.weight.val().dims().iter().product::<usize>()*4;
    let error=(after-before).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Нулей {zeros}/200; плотные байты {bytes}; MSE изменения {error:.6}");
    assert_eq!(zeros,100); assert_eq!(bytes,800); assert!(error>1e-5 && error<1.0);
}
