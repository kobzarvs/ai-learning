use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
fn main(){
    let device=device();

    // Для положительных Uniform(0,1) ELU(x)+1 равно x+1.
    let q=Tensor::<2>::random([12,8],Distribution::Uniform(0.0,1.0),&device)+1.0;
    let k=Tensor::<2>::random([12,8],Distribution::Uniform(0.0,1.0),&device)+1.0;
    let v=Tensor::<2>::random([12,8],Distribution::Uniform(0.0,1.0),&device);
    let mut memory=Tensor::<2>::zeros([8,8],&device); let mut normalizer=Tensor::<2>::zeros([1,8],&device);
    let mut recurrent=Vec::new(); let mut reference=Vec::new();
    for t in 0..12 {
        // Берём часть тензора по указанным диапазонам индексов.
        let qt=q.clone().slice([t..t+1]); let kt=k.clone().slice([t..t+1]);

        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        memory=memory+kt.clone().transpose().matmul(v.clone().slice([t..t+1])); normalizer=normalizer+kt;
        recurrent.push(qt.clone().matmul(memory.clone())/(qt.clone()*normalizer.clone()).sum_dim(1).add_scalar(1e-6));

        // Только префикс: тот же результат, что нижнетреугольная маска scores.
        let scores=qt.matmul(k.clone().slice([0..t+1]).transpose());
        reference.push(scores.clone().matmul(v.clone().slice([0..t+1]))/(scores.sum_dim(1)+1e-6));
    }

    // Читаем единственное значение тензора как обычное число Rust.
    let error=(Tensor::cat(recurrent,0)-Tensor::cat(reference,0)).abs().max().into_scalar::<f32>();
    println!("Максимальная разница: {error:.8}; память: 64+8 чисел"); assert!(error<1e-5);
}
