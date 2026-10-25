"""Application examples: trained Burn models and explicit Rust symbolic systems."""

def register(add):
    add('text lab-text', r'''
use burn::{nn::{LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,TensorData}};
fn encode(texts:&[&str],vocab:&[&str],d:&Device)->Tensor<2>{
 let mut values=vec![0.0f32;texts.len()*vocab.len()];
 for (i,text) in texts.iter().enumerate(){for word in text.split_whitespace(){if let Some(j)=vocab.iter().position(|w|*w==word){values[i*vocab.len()+j]+=1.0;}}}
 Tensor::from_data(TensorData::new(values,[texts.len(),vocab.len()]),d)
}
fn main(){
 let d=device().autodiff();
 let texts=["включи свет","свет включить","зажги лампу","лампу включить","выключи свет","свет выключить","погаси лампу","лампу выключить"];
 let mut vocab:Vec<_>=texts.iter().flat_map(|s|s.split_whitespace()).collect();vocab.sort();vocab.dedup();
 let x=encode(&texts,&vocab,&d);let y=Tensor::<1,Int>::from_data([1,1,1,1,0,0,0,0],&d);
 let mut model=LinearConfig::new(vocab.len(),2).init(&d);let ce=CrossEntropyLossConfig::new().init(&d);
 let before=ce.forward(model.forward(x.clone()),y.clone()).into_scalar::<f32>();
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();
 for _ in 0..160{let loss=ce.forward(model.forward(x.clone()),y.clone());let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);}
 let after=ce.forward(model.forward(x),y).into_scalar::<f32>();
 let checks=["включи лампу","выключи лампу","пожалуйста свет включить"];
 let prediction=model.forward(encode(&checks,&vocab,&d)).argmax(1).reshape([3]).into_data().try_into_vec_as::<i64>().unwrap();
 println!("Ошибка до {before:.4}, после {after:.4}; команды {prediction:?}");
 assert!(after<before*0.2);assert_eq!(prediction,vec![1,0,1]);
 // Неизвестное слово игнорируется; это ограничение представления, а не понимание отрицания.
 assert_eq!(encode(&["телепортируй"],&vocab,&d).sum().into_scalar::<f32>(),0.0);
 assert_eq!((encode(&["не включи свет"],&vocab,&d)-encode(&["включи свет"],&vocab,&d)).abs().max().into_scalar::<f32>(),0.0);
}
''','encode строит словарь только по восьми учебным фразам. Linear обучается 160 шагов AdamW; три новых сочетания проверяются отдельно. Неизвестные слова игнорируются.', 'Ошибка обучения уменьшается; новые команды получают [1,0,1]. Неизвестное слово даёт нулевые признаки; «не» не меняет представление.', exercise='Добавь «не» в vocab и отрицательные учебные фразы в texts и y. Обнови размеры y, число проверок и assert_eq!; проверь новые отрицания. Один подсчёт слов по-прежнему не учитывает порядок.')

    add('recommend',r'''
use burn::{module::{Module,Param},optim::{AdamWConfig,GradientsParams}};
#[derive(Module,Debug)]
struct Factors{users:Param<Tensor<2>>,items:Param<Tensor<2>>}
impl Factors{fn forward(&self)->Tensor<2>{self.users.val().matmul(self.items.val().transpose())}}
fn main(){
 let d=device().autodiff();
 let rating=Tensor::<2>::random([12,3],Distribution::Normal(0.0,1.0),&d).matmul(Tensor::<2>::random([10,3],Distribution::Normal(0.0,1.0),&d).transpose()).detach();
 let mask=Tensor::<2>::random([12,10],Distribution::Uniform(0.0,1.0),&d).lower_elem(0.8).float();let hidden=mask.clone().neg()+1.0;
 let known_count=mask.clone().sum();let hidden_count=hidden.clone().sum();assert!(hidden_count.clone().into_scalar::<f32>()>0.0);
 let mut model=Factors{users:Param::from_tensor(Tensor::<2>::random([12,3],Distribution::Normal(0.0,1.0),&d)),items:Param::from_tensor(Tensor::<2>::random([10,3],Distribution::Normal(0.0,1.0),&d))};
 let before=((model.forward()-rating.clone()).powf_scalar(2.0)*mask.clone()).sum()/known_count.clone();
 let baseline=(rating.clone().powf_scalar(2.0)*hidden.clone()).sum()/hidden_count.clone();
 let mut opt=AdamWConfig::new().with_weight_decay(0.01).init();
 for _ in 0..400{let loss=((model.forward()-rating.clone()).powf_scalar(2.0)*mask.clone()).sum()/known_count.clone();let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.04,model,grads);}
 let prediction=model.forward();let train=((prediction.clone()-rating.clone()).powf_scalar(2.0)*mask.clone()).sum()/known_count;
 let test=((prediction.clone()-rating).powf_scalar(2.0)*hidden.clone()).sum()/hidden_count;
 let a=before.into_scalar::<f32>();let b=train.into_scalar::<f32>();let c=test.into_scalar::<f32>();let z=baseline.into_scalar::<f32>();
 // Ранжируем только предметы, оценки которых не показывали при обучении.
 let scores=prediction.into_data().try_to_vec::<f32>().unwrap();let held=hidden.into_data().try_to_vec::<f32>().unwrap();
 for user in 0..12{let mut candidates:Vec<_>=(0..10).filter(|item|held[user*10+item]>0.5).collect();candidates.sort_by(|a,b|scores[user*10+b].total_cmp(&scores[user*10+a]));assert!(candidates.windows(2).all(|p|scores[user*10+p[0]]>=scores[user*10+p[1]]));println!("Пользователь {user}: скрытые кандидаты {candidates:?}");}
 println!("MSE известных до {a:.4}, после {b:.4}; скрытых {c:.4}; нулевой прогноз {z:.4}");assert!(b<a*0.1 && c.is_finite());
}
''','Factors обучает две матрицы [12,3] и [10,3] по 80% оценок, 400 шагов AdamW. Маска исключает скрытые оценки из функции ошибки. После обучения считаем их MSE и ранжируем скрытые предметы.', 'Ошибка на известных оценках снижается; отдельно показана ошибка скрытых сочетаний и порядок кандидатов для каждого пользователя.', exercise='Замени порог lower_elem(0.8) на 0.3. Сравни скрытую MSE с нулевым прогнозом z и проверь пользователей с малым числом известных оценок. При слабых данных ослабь assert для учебной ошибки, объяснив результат.')

    add('multimodal',r'''
use burn::{module::Module,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct Net{hidden:Linear,output:Linear}
impl Net{fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}}
fn accuracy(model:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32{model.forward(x).argmax(1).reshape([80]).equal(y).float().mean().into_scalar()}
fn main(){
 let d=device().autodiff();let image=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);let text=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);
 let y=(image.clone().slice([0..400,0..1])+text.clone().slice([0..400,0..1])).greater_elem(0.0).int().reshape([400]);let x=Tensor::cat(vec![image,text],1);
 let mut model=Net{hidden:LinearConfig::new(4,16).init(&d),output:LinearConfig::new(16,2).init(&d)};
 let ce=CrossEntropyLossConfig::new().init(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();
 let before=accuracy(&model,x.clone().slice([320..400]),y.clone().slice([320..400]));
 for _ in 0..160{let loss=ce.forward(model.forward(x.clone().slice([0..320])),y.clone().slice([0..320]));let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);}
 let test=x.slice([320..400]);let labels=y.slice([320..400]);let both=accuracy(&model,test.clone(),labels.clone());
 let missing=Tensor::cat(vec![test.slice([0..80,0..2]),Tensor::<2>::zeros([80,2],&d)],1);let one=accuracy(&model,missing,labels);
 println!("До {before:.3}; оба источника {both:.3}; без текста {one:.3}");assert!(both>0.9 && both>one);
}
''','Два источника по два признака объединяются в x. Net с SiLU обучается на первых 320 объектах; 80 новых сравниваются с теми же объектами при нулевом текстовом источнике.', 'На отложенных объектах точность с двумя источниками выше 0.9 и выше точности без второго источника.', exercise='В обучении создай маску [320,1] со случайными нулями и единицами и умножай на неё только столбцы 2..4. Сравни both и one после тех же 160 шагов; при изменении задачи пересмотри пороги assert.')

    add('rag lab-search',r'''
fn vector(text:&str,vocab:&[&str])->Vec<f64>{
 let mut v=vec![0.0;vocab.len()];for word in text.split_whitespace(){if let Some(i)=vocab.iter().position(|w|*w==word){v[i]+=1.0;}}
 let norm=v.iter().map(|x|x*x).sum::<f64>().sqrt();if norm>0.0{for x in &mut v{*x/=norm;}}v
}
fn retrieve<'a>(query:&str,documents:&'a [&str],vocab:&[&str])->Option<(usize,&'a str,f64)>{
 let q=vector(query,vocab);let mut best=None;let mut maximum=0.0;
 for (i,document) in documents.iter().enumerate(){let score=vector(document,vocab).iter().zip(&q).map(|(a,b)|a*b).sum::<f64>();if score>maximum{maximum=score;best=Some((i+1,*document,score));}}
 if maximum<0.2{None}else{best}
}
fn main(){
 let documents=["лимит загрузки файла 20 мегабайт","пароль можно изменить в настройках","экспорт заметок доступен в меню холста"];
 let mut vocab:Vec<_>=documents.iter().flat_map(|s|s.split_whitespace()).collect();vocab.sort();vocab.dedup();
 for (query,source) in [("какой лимит загрузки файла",1),("где изменить пароль",2),("экспорт заметок",3)]{
  let (id,fragment,score)=retrieve(query,&documents,&vocab).expect("Есть основание");assert_eq!(id,source);assert_eq!(fragment,documents[source-1]);println!("{query}: {fragment}. Источник: документ {id}; сходство {score:.3}");
 }
 for unknown in ["погода завтра","","квантовый двигатель"]{assert!(retrieve(unknown,&documents,&vocab).is_none());println!("{unknown:?}: В базе нет достаточного основания для ответа.");}
}
''','vector считает слова и нормирует длину вектора. retrieve сравнивает косинусное сходство с тремя документами, требует порог 0.2 и возвращает исходный фрагмент с номером источника. Это строковый алгоритм на Rust без обучения.', 'Три вопроса возвращают документы 1, 2 и 3 с точным фрагментом; неизвестные и пустой вопросы получают отказ.', exercise='Добавь в documents ещё один лимит с другой датой. Вместо одного best верни все документы выше порога и покажи обе даты: конфликт нельзя скрывать выбором первого фрагмента.')

    add('agents',r'''
#[derive(Debug,PartialEq)]
struct Fact{value:u32,unit:&'static str,source:&'static str}
fn lookup_limit(available:bool)->Result<Fact,&'static str>{
 if !available{return Err("Настройки проекта недоступны");}
 Ok(Fact{value:20,unit:"MB",source:"настройки проекта"})
}
fn run(plan:&[&str],available:bool)->Result<(String,usize),&'static str>{
 if plan.is_empty(){return Err("План пуст");}
 if plan.len()>3{return Err("Превышен лимит трёх шагов");}
 let mut observations=Vec::new();
 for name in plan{
  let fact=match *name{"lookup_limit"=>lookup_limit(available)?,_=>return Err("Неизвестный инструмент")};
  observations.push(fact);
 }
 let fact=observations.last().unwrap();Ok((format!("Лимит: {} {}. Источник: {}.",fact.value,fact.unit,fact.source),observations.len()))
}
fn main(){
 let (answer,calls)=run(&["lookup_limit"],true).unwrap();println!("{answer}");assert_eq!(calls,1);assert_eq!(answer,"Лимит: 20 MB. Источник: настройки проекта.");
 assert_eq!(run(&["shell"],true),Err("Неизвестный инструмент"));
 assert_eq!(run(&["lookup_limit"],false),Err("Настройки проекта недоступны"));
 assert_eq!(run(&[],true),Err("План пуст"));
 assert_eq!(run(&["lookup_limit";4],true),Err("Превышен лимит трёх шагов"));
 assert_eq!(run(&["lookup_limit","удали файлы"],true),Err("Неизвестный инструмент"));
 println!("Неизвестные действия, ошибка инструмента и превышение бюджета завершены без повторов.");
}
''','run проверяет бюджет плана, выбирает lookup_limit из разрешённых действий, сохраняет наблюдение и строит ответ из Fact. Ошибки возвращаются через Result без повторных вызовов. Нейросетевого планировщика здесь нет.', 'Ответ содержит 20 MB и источник «настройки проекта». Неизвестные инструменты, пустой план, недоступные настройки и превышение бюджета возвращают понятную ошибку.', exercise='Добавь в lookup_limit аргумент product и два продукта с разными лимитами. Передай название через шаг плана, проверь оба результата и неизвестный продукт. Сохрани ограничение run на три шага и отказ для неизвестного инструмента.')

    add('kg',r'''
type Fact<'a>=(&'a str,&'a str,&'a str);
fn locate<'a>(object:&str,facts:&[Fact<'a>])->Vec<(&'a str,Vec<Fact<'a>>)>{
 let mut answers=Vec::new();
 for &part in facts.iter().filter(|f|f.0==object && f.1=="часть"){
  for &location in facts.iter().filter(|f|f.0==part.2 && f.1=="находится"){
   answers.push((location.2,vec![part,location]));
  }
 }
 answers
}
fn main(){
 let facts=[("камера","часть","робот"),("робот","находится","цех"),("датчик","часть","станок"),("станок","находится","склад")];
 let camera=locate("камера",&facts);assert_eq!(camera,vec![("цех",vec![facts[0],facts[1]])]);
 let sensor=locate("датчик",&facts);assert_eq!(sensor,vec![("склад",vec![facts[2],facts[3]])]);
 assert!(locate("неизвестный объект",&facts).is_empty());assert!(locate("робот",&facts).is_empty());
 // Похожая связь с другим отношением не является основанием для этого запроса.
 let decoy=[("камера","рядом","станок"),("станок","находится","склад")];assert!(locate("камера",&decoy).is_empty());
 for (place,evidence) in camera{println!("Где находится камера: {place}. Основание: {evidence:?}");assert_eq!(evidence.len(),2);}
 println!("Датчик: {sensor:?}; неизвестный объект: нет основания.");
}
''','locate проходит строго по двум отношениям: «часть», затем «находится». Каждый ответ хранит ровно два использованных факта; посторонние отношения и неизвестные объекты не создают ответа.', 'Камера найдена в цехе, датчик на складе; основания содержат правильные пары фактов. Для неизвестного объекта и неправильного типа связи ответов нет.', exercise='Добавь дату в Fact и два факта местоположения робота. В locate выбирай самый поздний факт «находится», возвращай его дату с основанием и добавь assert для актуального места.')
