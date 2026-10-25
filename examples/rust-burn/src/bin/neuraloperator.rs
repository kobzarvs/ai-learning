use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Spectrum { real: Param<Tensor<2>>, imag: Param<Tensor<2>> }
impl Spectrum {
    fn forward(&self,x: Tensor<2>,cos: Tensor<2>,sin: Tensor<2>,inverse_cos: Tensor<2>,inverse_sin: Tensor<2>) -> Tensor<2> {
        // Матрично умножаем тензоры: внутренние размеры должны совпадать.
        let re = x.clone().matmul(cos); let im = x.matmul(sin).neg();
        let out_re = re.clone()*self.real.val()-im.clone()*self.imag.val();
        let out_im = re*self.imag.val()+im*self.real.val();
        out_re.matmul(inverse_cos)-out_im.matmul(inverse_sin)
    }
}
fn main() {
    // Включаем запись вычислений, чтобы затем получить градиенты ошибки.
    let device = device().autodiff();

    // Явный DFT той же длины 32: заменяет FFT, сохраняя комплексные коэффициенты.
    let mut c = Vec::new(); let mut s = Vec::new();
    for n in 0..32 { for k in 0..17 { let a = std::f32::consts::TAU*n as f32*k as f32/32.0; c.push(a.cos()); s.push(a.sin()); } }

    // Меняем форму тензора, сохраняя число и порядок элементов.
    let cos = Tensor::<1>::from_data(c.as_slice(),&device).reshape([32,17]);
    let sin = Tensor::<1>::from_data(s.as_slice(),&device).reshape([32,17]);
    let factors = Tensor::<1>::from_data((0..17).map(|k|if k==0 || k==16 {1.0_f32/32.0} else {2.0/32.0}).collect::<Vec<_>>().as_slice(),&device).reshape([17,1]);
    let ic = cos.clone().transpose()*factors.clone(); let is = sin.clone().transpose()*factors;
    let x = Tensor::<2>::random([180,32],Distribution::Normal(0.0,1.0),&device);
    let response = Tensor::<1>::from_data((0..17).map(|k|(-(k as f32)/4.0).exp()).collect::<Vec<_>>().as_slice(),&device).reshape([1,17]);

    // Отделяем тензор от графа градиентов, сохраняя его значения.
    let y = ((x.clone().matmul(cos.clone())*response.clone()).matmul(ic.clone())+(x.clone().matmul(sin.clone())*response).matmul(is.clone())).detach();

    // Регистрируем тензор как обучаемый параметр модели.
    let mut model = Spectrum {real:Param::from_tensor(Tensor::ones([1,17],&device)),imag:Param::from_tensor(Tensor::zeros([1,17],&device))};

    // Adam подбирает шаг обновления для каждого параметра по истории градиентов.
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();

    // Берём часть тензора по указанным диапазонам индексов.
    let eval = |m: &Spectrum| (m.forward(x.clone().slice([140..180]),cos.clone(),sin.clone(),ic.clone(),is.clone())-y.clone().slice([140..180])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before = eval(&model);
    for _ in 0..180 {
        let loss = (model.forward(x.clone().slice([0..140]),cos.clone(),sin.clone(),ic.clone(),is.clone())-y.clone().slice([0..140])).powf_scalar(2.0).mean();

        // Связываем градиенты вычислительного графа с параметрами этой модели.
        let grads = GradientsParams::from_grads(loss.backward(),&model); model = opt.step(0.05,model,grads);
    }
    let after = eval(&model); println!("Ошибка новых сигналов: {before} -> {after}");
    assert!(after<0.0001 && after<before*0.01);
}
