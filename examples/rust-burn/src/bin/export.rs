use burn::tensor::{Device, Distribution, Tensor};
fn device() -> Device {
    let device = Device::default();
    device.seed(7);
    println!("Burn 0.22.1 | устройство: {:?}", device);
    device
}
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
                // Матрично умножаем тензоры: внутренние размеры должны совпадать.
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

    // Читаем единственное значение тензора как обычное число Rust.
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
