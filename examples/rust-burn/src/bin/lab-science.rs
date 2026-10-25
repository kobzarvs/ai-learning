use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param}, optim::{AdamWConfig,GradientsParams}, tensor::activation::tanh};
#[derive(Module,Debug)]
struct Field { w: Param<Tensor<2>>, b: Param<Tensor<2>>, v: Param<Tensor<2>>, c: Param<Tensor<2>> }
impl Field {
    fn value_derivative(&self,x: Tensor<2>) -> (Tensor<2>,Tensor<2>) {
        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        let h = tanh(x.matmul(self.w.val())+self.b.val());
        let u = h.clone().matmul(self.v.val())+self.c.val();

        // Точная цепочка производных tanh: все операции остаются в графе Burn.
        let du = (h.powf_scalar(2.0).neg()+1.0)*self.w.val();
        (u,du.matmul(self.v.val()))
    }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();
    let mut model = Field {
        // Регистрируем тензор как обучаемый параметр модели.
        w: Param::from_tensor(Tensor::random([1,24],Distribution::Normal(0.0,0.4),&device)),
        b: Param::from_tensor(Tensor::zeros([1,24],&device)),
        v: Param::from_tensor(Tensor::random([24,1],Distribution::Normal(0.0,0.2),&device)),
        c: Param::from_tensor(Tensor::zeros([1,1],&device)),
    };
    let mut opt = AdamWConfig::new().with_weight_decay(0.01).with_epsilon(1e-8).init();
    for _ in 0..600 {
        let x = Tensor::<2>::random([80,1],Distribution::Uniform(0.0,2.0),&device);
        let (u,du) = model.value_derivative(x);
        let equation_error = (du+u).powf_scalar(2.0).mean();
        let boundary_error = (model.value_derivative(Tensor::zeros([1,1],&device)).0-1.0).powf_scalar(2.0).mean();
        let loss = equation_error+boundary_error;

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(),&model);
        model = opt.step(0.01,model,grads);
    }

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let x = Tensor::<1>::from_data((0..50).map(|i|2.0*i as f32/49.0).collect::<Vec<_>>().as_slice(),&device).reshape([50,1]);
    let (u,du) = model.value_derivative(x.clone());

    // Читаем единственное значение тензора как обычное число Rust.
    let residual = (du+u.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let error = (u-(-x).exp()).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка exp(-x): {error}; невязка: {residual}");
    assert!(error<0.002 && residual<0.005);
}
