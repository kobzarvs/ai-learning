"""Training strategies: executable parity with catalog.py, Burn 0.22.1."""
NET = r'''
#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
 fn new(inputs:usize,width:usize,device:&Device)->Self {Self{hidden:LinearConfig::new(inputs,width).init(device),output:LinearConfig::new(width,2).init(device)}}
 fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}
'''
POINTS = r'''
fn points(n:usize,d:&Device)->(Tensor<2>,Tensor<1,Int>){
 let x=Tensor::<2>::random([n,2],Distribution::Normal(0.0,1.0),d);
 let y=(x.clone().slice([0..n,0..1])+x.clone().slice([0..n,1..2])*0.5).greater_elem(0.0).int().reshape([n]); (x,y)
}
'''
def helpers(name, typ):
 return r'''
fn fit_NAME(mut model:TYPE,x:Tensor<2>,y:Tensor<1,Int>,steps:usize)->TYPE {
 let mut opt=AdamWConfig::new().with_weight_decay(0.001).init();
 let ce=CrossEntropyLossConfig::new().init(&x.device());
 for _ in 0..steps {let loss=ce.forward(model.forward(x.clone()),y.clone());let grads=GradientsParams::from_grads(loss.backward(),&model);model=opt.step(0.02,model,grads);} model
}
fn acc_NAME(model:&TYPE,x:Tensor<2>,y:Tensor<1,Int>)->f32 {model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()}
'''.replace('NAME',name).replace('TYPE',typ)
def base(*, net=False, linear=False, points=True, param=False, activations=(), linear_fit=True, hyper=False):
 modules = '{Module,Param}' if param and net else 'Module' if net else 'Param' if param else None
 imports = 'use burn::{' + ('module::'+modules+',' if modules else '')
 imports += 'nn::{Linear,LinearConfig,loss::CrossEntropyLossConfig},optim::{AdamWConfig,GradientsParams},tensor::'
 acts = (('silu',) if net else ()) + activations
 imports += ('{Int,activation::{'+','.join(acts)+'}}' if acts else 'Int') + '};\n'
 code = imports + (NET if net else '') + (POINTS if points else '')
 if net:
  fit = helpers('net','Net')
  if hyper:
   fit = fit.replace('steps:usize)', 'steps:usize,rate:f64,decay:f32)').replace('with_weight_decay(0.001)', 'with_weight_decay(decay)').replace('opt.step(0.02,', 'opt.step(rate,')
  code += fit
 if linear:
  parts = helpers('linear','Linear')
  code += parts if linear_fit else '\nfn acc_linear' + parts.split('\nfn acc_linear',1)[1]
 return code
def register(add):
 add('active',base(linear=True,activations=('softmax',))+r'''
fn main(){
 let d=device().autodiff();let (x,y)=points(320,&d);
 let first=fit_linear(LinearConfig::new(2,2).init(&d),x.clone().slice([0..20]),y.clone().slice([0..20]),160);
 let p=softmax(first.forward(x.clone().slice([20..240])),1).detach();
 let entropy=-(p.clone()*p.clamp_min(1e-8).log()).sum_dim(1).reshape([220]);
 let values=entropy.into_data().try_to_vec::<f32>().unwrap();
 let mut order:Vec<usize>=(0..220).collect();order.sort_by(|a,b|values[*b].total_cmp(&values[*a]));
 let chosen:Vec<i32>=order[..20].iter().map(|i|(*i+20) as i32).collect();
 assert!(chosen.iter().all(|i|*i>=20 && *i<240));
 let indices=Tensor::<1,Int>::from_ints(chosen.as_slice(),&d);
 // Случайный контроль: сортировка независимых случайных ключей без повторов.
 let keys=Tensor::<1>::random([220],Distribution::Uniform(0.0,1.0),&d).into_data().try_to_vec::<f32>().unwrap();
 let mut random:Vec<usize>=(0..220).collect();random.sort_by(|a,b|keys[*a].total_cmp(&keys[*b]));
 let ri:Vec<i32>=random[..20].iter().map(|i|(*i+20) as i32).collect();let ri=Tensor::<1,Int>::from_ints(ri.as_slice(),&d);
 let train=|ix:Tensor<1,Int>|(Tensor::cat(vec![x.clone().slice([0..20]),x.clone().select(0,ix.clone())],0),Tensor::cat(vec![y.clone().slice([0..20]),y.clone().select(0,ix)],0));
 let (ax,ay)=train(indices);let (rx,ry)=train(ri);
 let control=fit_linear(first.clone(),rx,ry,160);let active=fit_linear(first,ax,ay,160);
 let a=acc_linear(&active,x.clone().slice([240..320]),y.clone().slice([240..320]));let r=acc_linear(&control,x.slice([240..320]),y.slice([240..320]));
 println!("Запрошены {:?}; точность активного {a:.3}, случайного {r:.3}",chosen);assert!(a>0.8 && r>0.8);
}
''','Энтропия выбирает 20 наиболее неоднозначных точек из пула 20..240. Две копии одной обученной модели получают одинаковый бюджет меток; 80 тестовых точек не участвуют в выборе.','Обе модели классифицируют более 80% тестовых точек; активный выбор не обязан победить.',exercise='Повтори опыт с seed 1, 2 и 3 в device(). Сравни разброс двух точностей и номера запросов; преимущество активного выбора не добавляй в assert.')
 add('continual',base(net=True,points=False)+r'''
fn main(){
 let d=device().autodiff();let x=Tensor::<2>::random([400,2],Distribution::Normal(0.0,1.0),&d);
 let ya=x.clone().slice([0..400,0..1]).greater_elem(0.0).int().reshape([400]);let yb=x.clone().slice([0..400,1..2]).greater_elem(0.0).int().reshape([400]);
 let a=Tensor::cat(vec![x.clone(),Tensor::zeros([400,1],&d)],1);let b=Tensor::cat(vec![x,Tensor::ones([400,1],&d)],1);
 let first=fit_net(Net::new(3,24,&d),a.clone().slice([0..300]),ya.clone().slice([0..300]),160);
 let before=acc_net(&first,a.clone().slice([300..400]),ya.clone().slice([300..400]));
 let naive=fit_net(first.clone(),b.clone().slice([0..300]),yb.clone().slice([0..300]),160);
 let replay=fit_net(first,Tensor::cat(vec![b.clone().slice([0..300]),a.clone().slice([0..80])],0),Tensor::cat(vec![yb.clone().slice([0..300]),ya.clone().slice([0..80])],0),160);
 let old_naive=acc_net(&naive,a.clone().slice([300..400]),ya.clone().slice([300..400]));let old=acc_net(&replay,a.slice([300..400]),ya.slice([300..400]));
 let new_naive=acc_net(&naive,b.clone().slice([300..400]),yb.clone().slice([300..400]));let new=acc_net(&replay,b.slice([300..400]),yb.slice([300..400]));
 println!("Старая до {before:.3}, после {old_naive:.3}, с памятью {old:.3}; новая {new_naive:.3}, с памятью {new:.3}");
 assert!(before>0.9 && old>0.85 && new>0.85 && new_naive>0.9);assert!(old>old_naive+0.1);
}
''','Два задания различаются третьим признаком. После первой задачи клонируются веса: обычное дообучение сравнивается с памятью 80 старых примеров. Измеряются обе задачи на 100 новых точках.','Повторение сохраняет старую задачу лучше обычного дообучения; обе задачи у replay выше 85%.',exercise='Замени 80 старых примеров на 5, затем 20. При малой памяти убери assert!(old>old_naive+0.1), а порог old>0.85 замени проверкой old.is_finite(); сравни обе задачи.')
 add('federated',base(linear=True,param=True)+r'''
fn main(){
 let d=device().autodiff();let (x,y)=points(400,&d);let mut global=LinearConfig::new(2,2).init(&d);
 let before=acc_linear(&global,x.clone().slice([300..400]),y.clone().slice([300..400]));
 for _ in 0..15 {
 let a=fit_linear(global.clone(),x.clone().slice([0..150]),y.clone().slice([0..150]),5);
 let b=fit_linear(global.clone(),x.clone().slice([150..300]),y.clone().slice([150..300]),5);
 let w=(a.weight.val()+b.weight.val())*0.5;let bias=(a.bias.as_ref().unwrap().val()+b.bias.as_ref().unwrap().val())*0.5;
 global.weight=Param::from_tensor(w.clone().detach());global.bias=Some(Param::from_tensor(bias.clone().detach()));
 assert!((global.weight.val()-w).abs().max().into_scalar::<f32>()<1e-6);
 }
 let score=acc_linear(&global,x.slice([300..400]),y.slice([300..400]));println!("Общая модель: до {before:.3}, после {score:.3}");assert!(score>0.9);
}
''','15 раундов: два клиента начинают с одинаковых глобальных весов, обучаются по 5 шагов на своих 150 точках, сервер усредняет веса и bias. Тестовые 100 точек отделены.','Глобальная точность выше 90%; в каждом раунде проверяется арифметическое усреднение весов.','Локальная симуляция FedAvg без сети и защиты приватности. Равные размеры клиентов оправдывают равные веса.',exercise='Раздай клиентам разные категории через select по индексам меток. Сохрани 15 раундов и сравни score; при падении качества замени score>0.9 на score.is_finite(), сохрани проверку усреднения.')
 add('curriculum',base(net=True)+r'''
fn main(){
 let d=device().autodiff();let (x,y)=points(400,&d);
 let margin=(x.clone().slice([0..300,0..1])+x.clone().slice([0..300,1..2])*0.5).abs().reshape([300]).into_data().try_to_vec::<f32>().unwrap();
 let mut order:Vec<usize>=(0..300).collect();order.sort_by(|a,b|margin[*b].total_cmp(&margin[*a]));assert!(margin[order[0]]>=margin[order[299]]);
 let initial=Net::new(2,12,&d);let mut model=initial.clone();
 for size in [50,120,300] {let chosen:Vec<i32>=order[..size].iter().map(|i|*i as i32).collect();let ix=Tensor::<1,Int>::from_ints(chosen.as_slice(),&d);model=fit_net(model,x.clone().select(0,ix.clone()),y.clone().select(0,ix),50);println!("Объектов {size}: {:.3}",acc_net(&model,x.clone().slice([300..400]),y.clone().slice([300..400])));}
 // Те же три сброса AdamW и 150 обновлений, но все 300 точек с начала.
 let mut control=initial;for _ in 0..3 {control=fit_net(control,x.clone().slice([0..300]),y.clone().slice([0..300]),50);}
 let score=acc_net(&model,x.clone().slice([300..400]),y.clone().slice([300..400]));let baseline=acc_net(&control,x.slice([300..400]),y.slice([300..400]));println!("Учебный план {score:.3}, все сразу {baseline:.3}");assert!(score>0.9 && baseline>0.9);
}
''','Объекты упорядочены по убыванию расстояния до известной границы. Этапы 50/120/300 точек по 50 шагов сравниваются с теми же 150 обновлениями на полной учебной части.','Обе стратегии дают более 90% на 100 отложенных точках; преимущество порядка не гарантировано.','Формула простоты известна только в этой искусственной задаче.',exercise='В сортировке поменяй margin[*b].total_cmp(&margin[*a]) на margin[*a].total_cmp(&margin[*b]); разверни проверку первого и последнего margin. При добавлении шума меток замени пороги точности на is_finite() и отдельно измерь падение.')
 add('automl',base(net=True,hyper=True)+r'''
fn main(){
 let d=device().autodiff();let (x,y)=points(500,&d);let mut best:Option<Net>=None;let mut best_score=-1.0;let mut best_width=0;let mut scores=Vec::new();
 for width in [4,12,32] {d.seed(7);let model=fit_net(Net::new(2,width,&d),x.clone().slice([0..300]),y.clone().slice([0..300]),80,0.02,0.001);let score=acc_net(&model,x.clone().slice([300..400]),y.clone().slice([300..400]));scores.push(score);println!("Ширина {width}, валидация {score:.3}");if score>best_score {best_score=score;best_width=width;best=Some(model);}}
 assert!(scores.iter().all(|s|*s<=best_score));let score=acc_net(&best.unwrap(),x.slice([400..500]),y.slice([400..500]));println!("Выбрано {best_width}, валидация {best_score:.3}, тест {score:.3}");assert!(best_score>0.9 && score>0.9);
}
''','Три ширины 4/12/32 обучаются по 80 шагов с одинаковым seed. Выбор строго по валидации 300..400; тест 400..500 раскрывается только после выбора. При равенстве сохраняется первый кандидат, как max Python.','Выбран максимальный валидационный результат; окончательный тест выше 90%.',exercise='Расширь поиск: внутри цикла width добавь вложенные циклы for rate in [0.005,0.02] и for decay in [0.0,0.001]. Перед каждым кандидатом вызывай d.seed(7), передавай rate и decay последними аргументами fit_net вместо 0.02 и 0.001. Вместе с best_width сохраняй best_rate и best_decay. Выбирай по score на 300..400 и только один раз оцени победителя на 400..500. Получится 12 запусков по 80 шагов. Если слабый кандидат выигрывает из-за разброса, замени финальный score>0.9 на score>0.8, сохрани проверку максимума валидации.')
 add('distill',base(net=True,linear=True,linear_fit=False,activations=('softmax','log_softmax'))+r'''
fn main(){
 let d=device().autodiff();let (x,y)=points(400,&d);let teacher=fit_net(Net::new(2,32,&d),x.clone().slice([0..300]),y.clone().slice([0..300]),160);
 let teacher_w=teacher.hidden.weight.val().detach();let mut student=LinearConfig::new(2,2).init(&d);let mut opt=AdamWConfig::new().with_weight_decay(0.01).init();let ce=CrossEntropyLossConfig::new().init(&d);let temperature=2.0;
 let soft=softmax(teacher.forward(x.clone().slice([0..300]))/temperature,1).detach();
 for _ in 0..180 {let logits=student.forward(x.clone().slice([0..300]));let logp=log_softmax(logits.clone()/temperature,1);let kl=(soft.clone()*(soft.clone().clamp_min(1e-8).log()-logp)).sum()/300.0*(temperature*temperature);let loss=kl*0.7+ce.forward(logits,y.clone().slice([0..300]))*0.3;let grads=GradientsParams::from_grads(loss.backward(),&student);student=opt.step(0.03,student,grads);}
 let t=acc_net(&teacher,x.clone().slice([300..400]),y.clone().slice([300..400]));let s=acc_linear(&student,x.slice([300..400]),y.slice([300..400]));let teacher_params=teacher.num_params();let student_params=student.num_params();println!("Учитель {t:.3}, {teacher_params} параметров; ученик {s:.3}, {student_params} параметров");assert!(s>0.9 && t>0.9 && student_params<teacher_params);assert!((teacher.hidden.weight.val()-teacher_w).abs().max().into_scalar::<f32>()<1e-7);
}
''','Учитель 2→32→2 обучается на метках; линейный ученик получает 0.7 KL с температурой 2 и множителем T² плюс 0.3 cross entropy. Вероятности учителя отсоединены от графа; проверяется неизменность его весов.','На 100 новых точках обе точности выше 90%; ученик имеет 6 параметров против 162 у учителя.',exercise='Сразу после создания student сохрани let mut control=student.clone(). Создай отдельный AdamWConfig::new().with_weight_decay(0.01).init() для control и обучай его 180 шагов с opt.step(0.03,control,grads), вычисляя loss только через ce.forward(control.forward(x.clone().slice([0..300])),y.clone().slice([0..300])). Так обе модели начинают с одинаковых весов, имеют одинаковые rate, decay и число шагов; меняется только loss. Выведи точность control на тех же 100 тестовых точках. Не добавляй assert о превосходстве дистилляции.')
