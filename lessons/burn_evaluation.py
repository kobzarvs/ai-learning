"""Подготовка данных и честная оценка: самостоятельные программы Burn 0.22.1."""
CLASSIFY = r'''
use burn::{module::Module,nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::{Int,activation::silu}};
#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
 fn new(d:&Device)->Self {Self{hidden:LinearConfig::new(2,64).init(d),output:LinearConfig::new(64,2).init(d)}}
 fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}
fn fit(mut m:Net,x:Tensor<2>,y:Tensor<1,Int>)->Net {
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();let ce=CrossEntropyLossConfig::new().init(&x.device());
 for _ in 0..200 {let loss=ce.forward(m.forward(x.clone()),y.clone());let g=GradientsParams::from_grads(loss.backward(),&m);m=opt.step(0.02,m,g);}m
}
fn accuracy(m:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32 {let n=y.dims()[0];m.forward(x).argmax(1).reshape([n]).equal(y).float().mean().into_scalar()}
fn permutation(n:usize,d:&Device)->Tensor<1,Int>{
 let keys=Tensor::<1>::random([n],Distribution::Uniform(0.0,1.0),d).into_data().try_to_vec::<f32>().unwrap();
 let mut ix:Vec<usize>=(0..n).collect();ix.sort_by(|a,b|keys[*a].total_cmp(&keys[*b]));
 let ix:Vec<i32>=ix.into_iter().map(|i|i as i32).collect();Tensor::from_ints(ix.as_slice(),d)
}
'''

def register(add):
 add('data split',r'''
fn main(){
 let d=device();let raw=Tensor::<2>::random([300,3],Distribution::Normal(0.0,1.0),&d);
 let x=Tensor::cat(vec![raw.clone().slice([0..240]),raw.slice([240..300])+1.5],0);
 let train=x.clone().slice([0..180]);let validation=x.clone().slice([180..240]);let test=x.clone().slice([240..300]);
 let mean=train.clone().mean_dim(0);let centered=train.clone()-mean.clone();
 // torch.std использует поправку Бесселя: деление на n-1.
 let std=(centered.clone().powf_scalar(2.0).sum_dim(0)/179.0).sqrt().clamp_min(1e-6);
 let train_scaled=centered/std.clone();let test_scaled=(test.clone()-mean.clone())/std.clone();
 let leaked_mean=x.clone().mean_dim(0);let leaked_train=(train-leaked_mean.clone())/std.clone();
 println!("Формы {:?}/{:?}/{:?}; средние train {:?}, test {:?}; train при утечке {:?}",train_scaled.dims(),validation.dims(),test.dims(),train_scaled.clone().mean_dim(0).into_data(),test_scaled.clone().mean_dim(0).into_data(),leaked_train.clone().mean_dim(0).into_data());
 assert!(train_scaled.mean_dim(0).abs().max().into_scalar::<f32>()<1e-5);
 assert!((leaked_mean-mean).abs().max().into_scalar::<f32>()>0.1);
 assert!(test_scaled.mean().into_scalar::<f32>()>0.5);
 assert!(leaked_train.mean_dim(0).abs().max().into_scalar::<f32>()>0.1);
}
''','300 нормальных строк разделены 180/60/60. Последние 60 сдвинуты на 1.5. Среднее и стандартное отклонение с поправкой Бесселя вычисляются только по первым 180. Для сравнения отдельно показана утечка среднего всех строк.','Учебное среднее после честной нормализации около нуля; тест сохраняет сдвиг. Среднее по всем данным заметно сдвигает подготовленную учебную часть.',exercise='В mean замени train.clone() на x.clone(). Удали assert про нулевое среднее train_scaled и сравнение leaked_mean-mean (они теперь совпадают); сравни train_scaled с leaked_train; объясни, какие тестовые сведения попали в подготовку.')
 add('metrics',r'''
use burn::tensor::Int;
fn metrics(y:Tensor<1,Int>,p:Tensor<1,Int>)->(f32,f32,f32){
 let yt=y.clone().into_data().try_to_vec::<i32>().unwrap();let pt=p.clone().into_data().try_to_vec::<i32>().unwrap();
 let (mut tp,mut fp,mut fn_)=(0usize,0usize,0usize);
 for (a,b) in yt.iter().zip(pt.iter()){if *b==1 {if *a==1 {tp+=1}else{fp+=1}}else if *a==1 {fn_+=1}}
 (p.equal(y).float().mean().into_scalar(),tp as f32/(tp+fp).max(1) as f32,tp as f32/(tp+fn_).max(1) as f32)
}
fn main(){let d=device();let labels:Vec<i32>=(0..100).map(|i|if i>=95 {1}else{0}).collect();let y=Tensor::<1,Int>::from_ints(labels.as_slice(),&d);
 let baseline=metrics(y.clone(),Tensor::<1,Int>::zeros([100],&d));
 let alarms:Vec<i32>=(0..100).map(|i|if i<6 || (95..99).contains(&i){1}else{0}).collect();let alternative=metrics(y,Tensor::from_ints(alarms.as_slice(),&d));
 println!("Без тревог accuracy/precision/recall {:?}; 10 тревог, 4 события {:?}",baseline,alternative);
 assert!((baseline.0-0.95).abs()<1e-6 && baseline.1==0.0 && baseline.2==0.0);
 assert!((alternative.0-0.93).abs()<1e-6 && (alternative.1-0.4).abs()<1e-6 && (alternative.2-0.8).abs()<1e-6);
}
''','Тензорная accuracy и явный подсчёт TP/FP/FN на 95 обычных и 5 важных объектах. Второй контроль выдаёт 10 тревог: 4 настоящих и 6 ложных. При отсутствии тревог precision условно равен нулю.','Всегда ноль: 0.95/0/0. Десять тревог: accuracy 0.93, precision 0.4, recall 0.8.',exercise='Задай цену пропуска 20 и ложной тревоги 1. Вычисли стоимость обеих стратегий из FN и FP и сравни её с accuracy.')
 add('generalization',CLASSIFY+r'''
fn main(){let d=device().autodiff();let x=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);let y=(x.clone().slice([0..400,0..1])+x.clone().slice([0..400,1..2])*0.5).greater_elem(0.0).int().reshape([400]);
 let tx=x.clone().slice([0..300]);let ty=y.clone().slice([0..300]);let shuffled=ty.clone().select(0,permutation(300,&d));
 assert_eq!(ty.clone().sum().into_scalar::<i32>(),shuffled.clone().sum().into_scalar::<i32>());
 let changed=300-ty.clone().equal(shuffled.clone()).int().sum().into_scalar::<i32>();
 println!("Изменено учебных меток: {changed}/300");assert!(changed>60);
 let true_model=fit(Net::new(&d),tx.clone(),ty.clone());let random_model=fit(Net::new(&d),tx.clone(),shuffled.clone());
 let true_train=accuracy(&true_model,tx.clone(),ty);let random_train=accuracy(&random_model,tx,shuffled);
 let true_test=accuracy(&true_model,x.clone().slice([300..400]),y.clone().slice([300..400]));let random_test=accuracy(&random_model,x.slice([300..400]),y.slice([300..400]));
 println!("Правильные метки train/test {true_train:.3}/{true_test:.3}; перемешанные {random_train:.3}/{random_test:.3}");assert!(true_train>0.9 && true_test>0.9);assert!(true_test>random_test+0.15);assert!(random_train.is_finite());
}
''','Две сети 2→64→2 с SiLU действительно обучаются AdamW по 200 шагов: одна на правиле x₀+0.5x₁>0, другая на перестановке 300 учебных меток. Последние 100 точек оцениваются с настоящими метками.','Сеть с настоящими метками превосходит 90% на тесте и заметно превосходит контроль случайных меток. Учебная оценка случайных меток печатается отдельно.',exercise='Увеличь ширину 64 до 128 и число шагов 200 до 800. Сравни random_train и random_test: рост первой оценки не обязан улучшить вторую.')
 add('uncertainty',r'''
use burn::{module::Module,nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams},tensor::activation::tanh};
#[derive(Module,Debug)] struct Net {a:Linear,b:Linear}
impl Net {fn new(d:&Device)->Self{Self{a:LinearConfig::new(1,16).init(d),b:LinearConfig::new(16,1).init(d)}}fn forward(&self,x:Tensor<2>)->Tensor<2>{self.b.forward(tanh(self.a.forward(x)))}}
fn main(){let d=device().autodiff();let values:Vec<f32>=(0..80).map(|i|-1.0+2.0*i as f32/79.0).collect();let x=Tensor::<1>::from_floats(values.as_slice(),&d).reshape([80,1]);let y=x.clone().sin();let query=Tensor::<2>::from_floats([[-3.0],[0.0],[3.0]],&d);let mut answers=Vec::new();let mut losses=Vec::new();
 for seed in 0..5 {d.seed(seed);let mut m=Net::new(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.0).init();
 for _ in 0..200 {let loss=(m.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();let g=GradientsParams::from_grads(loss.backward(),&m);m=opt.step(0.02,m,g);}
 let loss=(m.forward(x.clone())-y.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();losses.push(loss);answers.push(m.forward(query.clone()).reshape([3]).into_data().try_to_vec::<f32>().unwrap());}
 for j in 0..3 {let mean=answers.iter().map(|a|a[j]).sum::<f32>()/5.0;let std=(answers.iter().map(|a|(a[j]-mean).powi(2)).sum::<f32>()/4.0).sqrt();println!("x={}: среднее {mean:.4}, разброс {std:.4}",[-3,0,3][j]);assert!(mean.is_finite() && std.is_finite());}
 println!("Ответы независимых обучений {:?}; MSE {:?}",answers,losses);assert!(losses.iter().all(|l|l.is_finite() && *l<0.02));
 // Не требуем ненулевой дисперсии: одинаковые seed в упражнении законно дают совпадение.
 println!("Согласие моделей не гарантирует правильность вне [-1,1]; разброс не является доверительным интервалом.");
}
''','Пять заново созданных сетей 1→16→1 с tanh обучаются по 200 шагов на sin(x), x∈[-1,1], с seed 0..4. Для -3/0/3 выводятся все ответы, среднее и выборочное стандартное отклонение.','Каждая сеть достигает учебной MSE ниже 0.02. Ответы и разброс конечны; порядок величин разброса не навязывается assertion.','Это чувствительность к инициализации, без калибровки интервалов и гарантии вне учебного диапазона.',exercise='Замени d.seed(seed) на d.seed(0), оставив создание Net внутри цикла. Сравни все ответы и разброс. Совпадение копий одного обучения не доказывает правильность.')
 add('explain',r'''
use burn::{nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::Int};
'''+CLASSIFY[CLASSIFY.index('fn permutation'):]+r'''
fn accuracy(m:&Linear,x:Tensor<2>,y:Tensor<1,Int>)->f32 {let n=y.dims()[0];m.forward(x).argmax(1).reshape([n]).equal(y).float().mean().into_scalar()}
fn main(){let d=device().autodiff();let x=Tensor::<2>::random([400,3],Distribution::Normal(0.0,1.0),&d);let y=(x.clone().slice([0..400,0..1])*2.0+x.clone().slice([0..400,1..2])).greater_elem(0.0).int().reshape([400]);
 let mut m=LinearConfig::new(3,2).init(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();let ce=CrossEntropyLossConfig::new().init(&d);
 for _ in 0..160 {let loss=ce.forward(m.forward(x.clone().slice([0..300])),y.clone().slice([0..300]));let g=GradientsParams::from_grads(loss.backward(),&m);m=opt.step(0.02,m,g);}
 let test=x.slice([300..400]);let labels=y.slice([300..400]);let baseline=accuracy(&m,test.clone(),labels.clone());let mut drops=Vec::new();
 for feature in 0..3 {let mut total=0.0;for _ in 0..5 {let column=test.clone().slice([0..100,feature..feature+1]).select(0,permutation(100,&d));let broken=test.clone().slice_assign([0..100,feature..feature+1],column);total+=baseline-accuracy(&m,broken,labels.clone());}drops.push(total/5.0);}
 println!("Тест {baseline:.3}, среднее падение по столбцам {:?}",drops);assert!(baseline>0.9);assert!(drops.iter().all(|v|v.is_finite()));assert!(drops[0]>drops[1] && drops[0]>drops[2]+0.15);
}
''','Линейная модель обучается по 300 строкам с правилом 2x₀+x₁>0. Важность измеряется только на 100 отложенных строках: пять независимых перестановок каждого столбца, остальные столбцы и метки сохраняются.','Тестовая точность выше 90%. Первый признак имеет наибольшее среднее падение качества; третий не участвует в истинной разметке.','Перестановочная важность описывает зависимость обученной модели на выбранных данных и не доказывает причинность.',exercise='Перед let y вставь две строки: let first=x.clone().slice([0..400,0..1]); let x=x.slice_assign([0..400,2..3],first); Это делает третий столбец копией первого без ошибки владения. Сохрани тестовое измерение; убери assertion с ранжированием drops[0] и сравни важность коррелирующих признаков.')
