"""Compression, device execution and model persistence experiments for Burn 0.22.1."""

TRAIN = r'''
use burn::{nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams}};
fn fit(mut model:Linear,x:Tensor<2>,y:Tensor<2>,steps:usize)->Linear {
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.0).init();
    for _ in 0..steps {
        let loss=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&model);
        model=optimizer.step(0.02,model,grads);
    }
    model
}
'''


def register(add):
    add('quant', r'''
use burn::tensor::TensorData;
fn main(){
    let device=device();
    let weights=Tensor::<2>::random([128,64],Distribution::Normal(0.0,0.1),&device);
    let scale=weights.clone().abs().max_dim(1).clamp_min(1e-8)/127.0;
    let rounded=(weights.clone()/scale.clone()).round().clamp(-127.0,127.0);
    // Реальный буфер int8: не считаем float-тензор с целыми значениями сжатым.
    let packed:Vec<i8>=rounded.into_data().try_to_vec::<f32>().unwrap().into_iter().map(|v|v as i8).collect();
    let restored=Tensor::<2>::from_data(TensorData::new(packed.iter().map(|v|*v as f32).collect::<Vec<_>>(),[128,64]),&device)*scale.clone();
    let ratio=((restored.clone()-weights.clone()).abs()/scale).max().into_scalar::<f32>();
    let x=Tensor::<2>::random([20,64],Distribution::Normal(0.0,1.0),&device);
    let error=(x.clone().matmul(restored.transpose())-x.matmul(weights.transpose())).powf_scalar(2.0).mean().into_scalar::<f32>();
    let before=128*64*4; let after=packed.len()+128*4;
    println!("MSE выхода {error:.8}; байт до {before}, после с масштабами {after}");
    assert_eq!(after,8704); assert!(after<before/3);
    assert!(ratio<=0.501 && error>0.0 && error<0.001);
}
''', 'Burn вычисляет отдельный масштаб максимального модуля для каждой из 128 строк. Округлённые числа действительно копируются в Vec<i8>; при восстановлении они возвращаются на выбранное устройство. Проверяются ошибка округления в половину шага и MSE матричного произведения.', '8192 весов и 128 масштабов занимают 8704 байта вместо 32768; MSE выхода меньше 0.001.', limits='Измеряется полезная нагрузка буферов без служебных объектов. Восстановленное умножение выполняется float32 и не доказывает ускорение int8-ядер.', exercise='Перед расчётом scale замени один вес через slice_assign([0..1,0..1],Tensor::<2>::ones([1,1],&device)*10.0). Сравни текущий масштаб строк с weights.clone().abs().max().reshape([1,1])/127.0. Для ветви с общим масштабом замени after=packed.len()+128*4 на after=packed.len()+4 и assert_eq!(after,8704) на assert_eq!(after,8196): теперь хранится один float32-масштаб. Для исходной ветви с отдельными масштабами сохрани 8704 байта. Выведи обе MSE, убрав порог error<0.001 для опыта с выбросом; проверку ratio<=0.501 сохрани для масштаба каждой ветви.')

    add('pruning', r'''
use burn::{nn::LinearConfig,module::Param};
fn main(){
    let device=device(); let mut model=LinearConfig::new(20,10).init(&device);
    let x=Tensor::<2>::random([30,20],Distribution::Normal(0.0,1.0),&device);
    let before=model.forward(x.clone()); let weights=model.weight.val();
    let mut sorted=weights.clone().abs().into_data().try_to_vec::<f32>().unwrap();
    sorted.sort_by(|a,b|a.total_cmp(b));
    // torch.quantile(..., 0.5): линейная интерполяция двух центральных значений.
    let threshold=(sorted[99]+sorted[100])*0.5;
    model.weight=Param::from_tensor(weights.clone()*weights.abs().greater_equal_elem(threshold).float());
    let after=model.forward(x);
    let zeros=model.weight.val().equal_elem(0.0).int().sum().into_scalar::<i32>();
    let bytes=model.weight.val().dims().iter().product::<usize>()*4;
    let error=(after-before).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Нулей {zeros}/200; плотные байты {bytes}; MSE изменения {error:.6}");
    assert_eq!(zeros,100); assert_eq!(bytes,800); assert!(error>1e-5 && error<1.0);
}
''', 'Для матрицы Burn Linear размера 20×10 медиана модулей вычисляется с той же интерполяцией, что quantile в Python. Маска обнуляет малые веса, bias сохраняется; реальные ответы слоя сравниваются на 30 строках.', 'Ровно 100 из 200 случайных весов обнулены; плотная матрица по-прежнему содержит 800 байт, ответы изменяются.', exercise='Создай второй LinearConfig::new(20,5).init(&device). Перенеси первые пять столбцов model.weight.val().slice([0..20,0..5]) и первые пять элементов bias через Param::from_tensor. Сравни его выход с after.slice([0..30,0..5]); для этого сохрани after.clone() при расчёте error. Посчитай число параметров и байты с bias у обоих слоёв.')

    add('binary', r'''
fn main(){
    let device=device();
    let weights=Tensor::<2>::random([32,16],Distribution::Normal(0.0,1.0),&device);
    let scale=weights.clone().abs().mean_dim(1);
    let binary=(weights.clone().greater_equal_elem(0.0).float()*2.0-1.0)*scale.clone();
    let x=Tensor::<2>::random([40,16],Distribution::Normal(0.0,1.0),&device);
    let error=(x.clone().matmul(binary.clone().transpose())-x.matmul(weights.transpose())).powf_scalar(2.0).mean().into_scalar::<f32>();
    let magnitude_error=(binary.abs()-scale).abs().max().into_scalar::<f32>();
    println!("MSE выхода {error:.5}; теоретические биты знаков 512; фактические байты float32 2048");
    assert_eq!(magnitude_error,0.0); assert!(error>0.1 && error<30.0);
}
''', 'Каждый вес заменяется знаком ±1, умноженным на средний модуль его строки. Burn выполняет оба матричных произведения. Проверяется, что все восстановленные модули точно равны масштабу строки и выход заметно изменился.', '512 знаков теоретически требуют 512 бит плюс масштабы; здесь бинаризованный float32-тензор занимает 2048 байт, MSE заметна.', exercise='До потребления weights сохрани weights.clone(). Для тех же весов и x добавь квантование со scale8=weights.clone().abs().max_dim(1)/127.0 и восстановление (weights.clone()/scale8.clone()).round().clamp(-127.0,127.0)*scale8. Сохрани x.clone() в исходном произведении и сравни обе MSE.')

    add('mac', TRAIN+r'''
fn main(){
    let device=device().autodiff();
    let x=Tensor::<2>::random([64,2],Distribution::Normal(0.0,1.0),&device);
    let y=x.clone().sum_dim(1); let model=LinearConfig::new(2,1).init(&device);
    let before=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let model=fit(model,x.clone(),y.clone(),150);
    let after=(model.forward(x)-y).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Выбранное устройство {:?}; MSE до {before:.6}, после {after:.6}",device);
    assert!(after<0.01 && after<before*0.02);
}
''', 'Linear, случайные данные, градиенты и AdamW работают на одном выбранном Burn устройстве. 150 обновлений учат сумму двух координат. into_scalar дожидается результата перед выводом ошибки.', 'На выбранном устройстве MSE после 150 шагов меньше 0.01 и 2% исходной ошибки.', limits='CPU Flex, CUDA и Metal выбираются общим механизмом запуска и доступными Cargo features. Это проверка исполнения, а не сравнение быстродействия библиотек.', exercise='Измени число строк 64 на 65536 в random. Перед fit создай let start=std::time::Instant::now(); после вычисления after выведи start.elapsed(). into_scalar при расчёте after синхронизирует получение результата до остановки таймера. Сравни оба размера на одном устройстве.')

    add('repro', TRAIN+r'''
fn run(seed:u64,x:Tensor<2>,y:Tensor<2>,device:&Device)->Tensor<2>{
    device.seed(seed);
    let model=fit(LinearConfig::new(2,1).init(device),x.clone(),y,20);
    model.forward(x).detach()
}
fn main(){
    let device=device().autodiff();
    let x=Tensor::<2>::random([100,2],Distribution::Normal(0.0,1.0),&device);
    let y=x.clone().sum_dim(1);
    let a=run(7,x.clone(),y.clone(),&device);
    let b=run(7,x.clone(),y.clone(),&device);
    let c=run(8,x,y,&device);
    let same=(a.clone()-b).abs().max().into_scalar::<f32>();
    let different=(a-c).abs().max().into_scalar::<f32>();
    println!("Burn 0.22.1; одинаковый seed: max diff {same}; другой seed: {different}");
    assert_eq!(same,0.0); assert!(different>1e-4);
}
''', 'Данные создаются один раз. run сбрасывает генератор перед созданием Linear и выполняет ровно 20 шагов AdamW. Два запуска с seed 7 сравниваются побитово через нулевую разность; seed 8 должен дать другой ответ короткого обучения.', 'На одном устройстве и в том же окружении seed 7 повторяет ответы; seed 8 меняет результат.', limits='Побитовое совпадение между разными устройствами, версиями и реализациями ядер не обещается.', exercise='В run измени число шагов fit с 20 на 300. Выведи different и сравни с коротким обучением. Замени assert!(different>1e-4) на assert!(different<0.02): при сходимости разница может стать практически нулевой.')

    add('export', r'''
use burn::tensor::{TensorData,activation::silu};
use std::{fs,path::Path};
const MAGIC:&[u8]=b"ATLAS-GRAPH-v1\n";
// Последовательный граф: opcode 1 = Linear с размерами и параметрами, 2 = SiLU.
enum Node { Linear { weight:Tensor<2>,bias:Tensor<2> }, Silu }
struct Graph { input:usize,output:usize,nodes:Vec<Node> }
struct Reader<'a> { bytes:&'a [u8],offset:usize }
impl<'a> Reader<'a> {
    fn take<const N:usize>(&mut self)->Result<[u8;N],String> {
        let end=self.offset.checked_add(N).ok_or("Переполнение смещения")?;
        let bytes=self.bytes.get(self.offset..end).ok_or("Файл оборван")?;
        self.offset=end; Ok(bytes.try_into().unwrap())
    }
    fn dimension(&mut self)->Result<usize,String> {
        let n=u32::from_le_bytes(self.take()?) as usize;
        if n==0 || n>1024 {return Err("Недопустимый размер".into());} Ok(n)
    }
    fn values(&mut self,n:usize)->Result<Vec<f32>,String> {
        if n>65536 {return Err("Слишком много параметров узла".into());}
        let mut values=Vec::with_capacity(n);
        for _ in 0..n {
            let value=f32::from_le_bytes(self.take()?);
            if !value.is_finite() {return Err("Неконечный параметр".into());}
            values.push(value);
        }
        Ok(values)
    }
}
impl Graph {
    fn decode(bytes:&[u8],device:&Device)->Result<Self,String> {
        if bytes.len()>1_048_576 || !bytes.starts_with(MAGIC) {return Err("Неизвестный формат или размер файла".into());}
        let mut reader=Reader {bytes,offset:MAGIC.len()};
        let input=reader.dimension()?; let count=u32::from_le_bytes(reader.take()?) as usize;
        if count==0 || count>16 {return Err("Недопустимое число узлов".into());}
        let mut width=input; let mut nodes=Vec::new();
        for _ in 0..count {
            match reader.take::<1>()?[0] {
                1=>{
                    let inputs=reader.dimension()?; let outputs=reader.dimension()?;
                    if inputs!=width {return Err("Несогласованные размеры узлов".into());}
                    let weights=reader.values(inputs.checked_mul(outputs).ok_or("Переполнение размера")?)?;
                    let bias=reader.values(outputs)?;
                    nodes.push(Node::Linear {
                        weight:Tensor::from_data(TensorData::new(weights,[inputs,outputs]),device),
                        bias:Tensor::from_data(TensorData::new(bias,[1,outputs]),device),
                    }); width=outputs;
                },
                2=>nodes.push(Node::Silu),
                _=>return Err("Неизвестная операция".into()),
            }
        }
        if reader.offset!=bytes.len() {return Err("Лишние данные после графа".into());}
        Ok(Self {input,output:width,nodes})
    }
    fn load(path:&Path,device:&Device)->Result<Self,String> {
        if fs::metadata(path).map_err(|e|e.to_string())?.len()>1_048_576 {return Err("Слишком большой файл".into());}
        Self::decode(&fs::read(path).map_err(|e|e.to_string())?,device)
    }
    fn run(&self,mut x:Tensor<2>)->Tensor<2> {
        assert_eq!(x.dims()[1],self.input);
        for node in &self.nodes {
            x=match node {
                Node::Linear {weight,bias}=>x.matmul(weight.clone())+bias.clone(),
                Node::Silu=>silu(x),
            };
        }
        assert_eq!(x.dims()[1],self.output); x
    }
}
fn integer(bytes:&mut Vec<u8>,n:usize){bytes.extend_from_slice(&(n as u32).to_le_bytes());}
fn linear(bytes:&mut Vec<u8>,weight:&Tensor<2>,bias:&Tensor<2>){
    bytes.push(1); let [inputs,outputs]=weight.dims();
    integer(bytes,inputs); integer(bytes,outputs); assert_eq!(bias.dims(),[1,outputs]);
    for tensor in [weight,bias] {
        for value in tensor.clone().into_data().try_to_vec::<f32>().unwrap(){
            assert!(value.is_finite()); bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}
struct Model { w1:Tensor<2>,b1:Tensor<2>,w2:Tensor<2>,b2:Tensor<2>,activation:bool }
impl Model {
    fn new(hidden:usize,activation:bool,device:&Device)->Self {
        Self {
            w1:Tensor::random([4,hidden],Distribution::Uniform(-0.5,0.5),device),
            b1:Tensor::random([1,hidden],Distribution::Uniform(-0.5,0.5),device),
            w2:Tensor::random([hidden,2],Distribution::Uniform(-0.35,0.35),device),
            b2:Tensor::random([1,2],Distribution::Uniform(-0.35,0.35),device),
            activation,
        }
    }
    fn forward(&self,x:Tensor<2>)->Tensor<2>{
        let hidden=x.matmul(self.w1.clone())+self.b1.clone();
        let hidden=if self.activation {silu(hidden)} else {hidden};
        hidden.matmul(self.w2.clone())+self.b2.clone()
    }
    fn export(&self,path:&Path){
        let mut bytes=MAGIC.to_vec(); integer(&mut bytes,self.w1.dims()[0]);
        integer(&mut bytes,if self.activation {3} else {2});
        linear(&mut bytes,&self.w1,&self.b1);
        if self.activation {bytes.push(2);}
        linear(&mut bytes,&self.w2,&self.b2); fs::write(path,bytes).unwrap();
    }
}
fn roundtrip(hidden:usize,activation:bool,path:&Path,device:&Device){
    let model=Model::new(hidden,activation,device);
    let x3=Tensor::<2>::random([3,4],Distribution::Normal(0.0,1.0),device);
    let x5=Tensor::<2>::random([5,4],Distribution::Normal(0.0,1.0),device);
    let original3=model.forward(x3.clone()); let original5=model.forward(x5.clone());
    model.export(path); drop(model);
    // После чтения используются только узлы Graph; вызовов Model::forward здесь нет.
    let loaded=Graph::load(path,device).unwrap();
    assert_eq!(loaded.nodes.len(),if activation {3} else {2});
    match &loaded.nodes[0] {Node::Linear {weight,..}=>assert_eq!(weight.dims(),[4,hidden]),_=>panic!("Ожидался Linear")};
    let error3=(original3-loaded.run(x3)).abs().max().into_scalar::<f32>();
    let error5=(original5-loaded.run(x5)).abs().max().into_scalar::<f32>();
    assert!(error3<1e-6 && error5<1e-6);
    let bytes=fs::read(path).unwrap();
    assert!(Graph::decode(b"bad",device).is_err());
    assert!(Graph::decode(&bytes[..bytes.len()-1],device).is_err());
    let mut extra=bytes.clone(); extra.push(0); assert!(Graph::decode(&extra,device).is_err());
    let opcode=MAGIC.len()+8;
    let mut unknown=bytes.clone(); unknown[opcode]=255; assert!(Graph::decode(&unknown,device).is_err());
    let mut inconsistent=bytes.clone(); inconsistent[opcode+1..opcode+5].copy_from_slice(&3_u32.to_le_bytes());
    assert!(Graph::decode(&inconsistent,device).is_err());
    let mut nonfinite=bytes; nonfinite[opcode+9..opcode+13].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(Graph::decode(&nonfinite,device).is_err());
    println!("Файл: 4→{hidden}→2, SiLU={activation}, max diff пакетов 3/5: {error3}/{error5}");
}
fn main(){
    let device=device();
    let path=std::env::temp_dir().join(format!("atlas-burn-export-{}.bin",std::process::id()));
    for (hidden,activation) in [(8,true),(5,true),(5,false)] {roundtrip(hidden,activation,&path,&device);}
    fs::remove_file(path).unwrap();
}
''', 'Model::export записывает последовательный граф: число входов, список операций Linear/SiLU, размеры каждого Linear и его веса с bias в float32 little-endian. Независимый Graph::load восстанавливает операции из файла; Graph::run выполняет их общим циклом после уничтожения Model. Проверяются исходная схема 4→8→SiLU→2, ширина 5 и вариант без SiLU на двух новых пакетах. Повреждённые файлы отклоняются.', 'Для всех трёх топологий ответы Graph на пакетах 3×4 и 5×4 совпадают с Model с максимальной разницей меньше 1e-6. Загрузчик отвергает неизвестный формат/операцию, обрыв, лишние байты, NaN и несогласованные размеры.', limits='Явный учебный экспорт поддерживает только последовательность Linear и SiLU, до 16 узлов, размерности до 1024 и до 65536 весов на узел; файл ограничен 1 МиБ. Это не автоматический захват произвольного Rust/PyTorch-кода и не переносимый стандарт для телефона.', exercise='В список вариантов main добавь (12,false) и (12,true). Тот же Graph::load и Graph::run должен прочитать обе новые схемы без изменений. В roundtrip добавь ещё пакет x7 формы [7,4], сохрани model.forward(x7.clone()) до drop(model), затем сравни с loaded.run(x7) с допуском 1e-6. Сохрани все проверки повреждённых файлов.')
