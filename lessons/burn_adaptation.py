"""Adaptation tasks matching the executable PyTorch lessons (Burn 0.22.1)."""

NET = r'''
#[derive(Module,Debug)]
struct Net { hidden:Linear, output:Linear }
impl Net {
    fn new(inputs:usize,width:usize,outputs:usize,device:&Device)->Self {
        Self{hidden:LinearConfig::new(inputs,width).init(device),output:LinearConfig::new(width,outputs).init(device)}
    }
    fn forward(&self,x:Tensor<2>)->Tensor<2>{self.output.forward(silu(self.hidden.forward(x)))}
}
'''

POINTS = r'''
fn points(n:usize,device:&Device)->(Tensor<2>,Tensor<1,Int>){
    let x=Tensor::<2>::random([n,2],Distribution::Normal(0.0,1.0),device);
    let y=(x.clone().slice([0..n,0..1])+x.clone().slice([0..n,1..2])*0.5).greater_elem(0.0).int().reshape([n]);
    (x,y)
}
'''

ACCURACY = r'''
fn accuracy(model:&Net,x:Tensor<2>,y:Tensor<1,Int>)->f32 {
    model.forward(x).argmax(1).reshape([y.dims()[0]]).equal(y).float().mean().into_scalar()
}
'''

FIT = r'''
fn fit(mut model:Net,x:Tensor<2>,y:Tensor<1,Int>)->Net {
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.001).init();
    let ce=CrossEntropyLossConfig::new().init(&x.device());
    for _ in 0..160 {
        let loss=ce.forward(model.forward(x.clone()),y.clone());
        let grads=GradientsParams::from_grads(loss.backward(),&model);
        model=optimizer.step(0.02,model,grads);
    }
    model
}
'''

def network(*, classification=False, points=False, accuracy=False, softmax=False):
    imports = 'use burn::{module::Module, nn::{Linear,LinearConfig'
    if classification:
        imports += ',loss::CrossEntropyLossConfig'
    imports += '}, optim::{AdamWConfig,GradientsParams},tensor::'
    if classification or points or accuracy:
        imports += '{Int,activation::{silu'
        if softmax:
            imports += ',softmax'
        imports += '}}};\n'
    else:
        imports += 'activation::silu};\n'
    return imports + NET + (POINTS if points else '') + (ACCURACY if accuracy else '') + (FIT if classification else '')

def register(add):
    add('selfsupervised masked', network()+r'''
fn main(){
    let device=device().autodiff();
    let t=Tensor::<2>::random([320,1],Distribution::Normal(0.0,1.0),&device);
    let x=Tensor::cat(vec![t.clone(),t.clone()*2.0,-t.clone(),t*0.5],1)+Tensor::<2>::random([320,4],Distribution::Normal(0.0,0.03),&device);
    let mut model=Net::new(8,24,4,&device);
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.01).init();
    let test=x.clone().slice([250..320]);
    let mask=Tensor::<2>::zeros([70,4],&device).slice_assign([0..70,1..2],Tensor::<2>::ones([70,1],&device));
    let probe=Tensor::cat(vec![test.clone()*(1.0-mask.clone()),mask],1);
    let target=test.slice([0..70,1..2]);
    let before=(model.forward(probe.clone()).slice([0..70,1..2])-target.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    for _ in 0..220 {
        let mask=Tensor::<2>::random([250,4],Distribution::Uniform(0.0,1.0),&device).lower_elem(0.35).float();
        let train=x.clone().slice([0..250]);
        let observed=train.clone()*(1.0-mask.clone());
        let prediction=model.forward(Tensor::cat(vec![observed,mask.clone()],1));
        let loss=((prediction-train).powf_scalar(2.0)*mask.clone()).sum()/mask.sum().clamp_min(1.0);
        let grads=GradientsParams::from_grads(loss.backward(),&model);
        model=optimizer.step(0.02,model,grads);
    }
    let error=(model.forward(probe).slice([0..70,1..2])-target).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE скрытого столбца: до {before:.5}, после {error:.5}");
    assert!(error<0.1 && error<before*0.25);
}
''', 'Net получает четыре наблюдения и четыре числа маски. Случайная маска скрывает 35% элементов 250 учебных строк; ошибка суммируется только по скрытым элементам. Последние 70 строк проверяют восстановление второго столбца.', 'MSE второго скрытого столбца меньше 0.1 и четверти исходной ошибки.', exercise='Замени построение x на Tensor::<2>::random([320,4],Distribution::Normal(0.0,1.0),&device). Повтори обучение и объясни, почему проверочное assert перестаёт выполняться при независимых признаках.')

    add('semisupervised', network(classification=True,points=True,accuracy=True,softmax=True)+r'''
fn main(){
    let device=device().autodiff(); let (x,y)=points(500,&device);
    let model=fit(Net::new(2,16,2,&device),x.clone().slice([0..30]),y.clone().slice([0..30]));
    let before=accuracy(&model,x.clone().slice([400..500]),y.clone().slice([400..500]));
    // Один и тот же экземпляр сначала служит учителем, затем дообучается, как в Python.
    let probabilities=softmax(model.forward(x.clone().slice([30..400])).detach(),1);
    let (confidence,pseudo)=probabilities.max_dim_with_indices(1);
    let values=confidence.reshape([370]).into_data().try_to_vec::<f32>().unwrap();
    let chosen:Vec<i32>=values.iter().enumerate().filter(|(_,v)|**v>0.95).map(|(i,_)|i as i32).collect();
    assert!(!chosen.is_empty(),"Уверенные псевдометки должны существовать");
    let indices=Tensor::<1,Int>::from_ints(chosen.as_slice(),&device);
    let ax=Tensor::cat(vec![x.clone().slice([0..30]),x.clone().slice([30..400]).select(0,indices.clone())],0);
    let ay=Tensor::cat(vec![y.clone().slice([0..30]),pseudo.reshape([370]).select(0,indices)],0);
    let model=fit(model,ax,ay);
    let after=accuracy(&model,x.slice([400..500]),y.slice([400..500]));
    println!("Псевдометок {}; точность до {before:.3}, после {after:.3}",chosen.len());
    assert!(before>0.8 && after>0.8,"Псевдометки не гарантируют улучшения");
}
''', '30 размеченных точек обучают учителя. max_dim_with_indices выбирает класс и уверенность; только ответы выше 0.95 добавляются через select к обучению той же сети. Метки последних 100 точек используются лишь для оценки.', 'Есть уверенные псевдометки; точность на отложенных точках выше 0.8. Улучшение после дообучения не гарантировано.', exercise='В filter измени порог 0.95 на 0.6 и 0.99. Сравни chosen.len() и обе точности при том же device.seed(7).')

    add('transfer domain', network(classification=True,points=True)+r'''
fn main(){
    let device=device().autodiff(); let (x,y)=points(400,&device);
    let source=fit(Net::new(2,16,2,&device),x.clone().slice([0..300]),y.slice([0..300]));
    let encoder=source.hidden;
    let old_weight=encoder.weight.val().detach();
    let old_bias=encoder.bias.as_ref().unwrap().val().detach();
    // Отдельная голова — единственный Module, переданный optimizer.step.
    let features=silu(encoder.forward(x.clone())).detach();
    let labels=(x.clone().slice([0..400,0..1])-x.slice([0..400,1..2])*0.3).greater_elem(0.2).int().reshape([400]);
    let mut head=LinearConfig::new(16,2).init(&device);
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.001).init();
    let ce=CrossEntropyLossConfig::new().init(&device);
    for _ in 0..160 {
        let loss=ce.forward(head.forward(features.clone().slice([0..60])),labels.clone().slice([0..60]));
        let grads=GradientsParams::from_grads(loss.backward(),&head); head=optimizer.step(0.02,head,grads);
    }
    let accuracy=head.forward(features.slice([300..400])).argmax(1).reshape([100]).equal(labels.slice([300..400])).float().mean().into_scalar::<f32>();
    let drift=(encoder.weight.val()-old_weight).abs().max().into_scalar::<f32>()+(encoder.bias.as_ref().unwrap().val()-old_bias).abs().max().into_scalar::<f32>();
    println!("Точность новой головы {accuracy:.3}; обучаемых чисел 34; изменение кодировщика {drift}");
    assert_eq!(drift,0.0); assert!(accuracy>0.85);
}
''', 'Исходная сеть обучается на 300 точках. Кодировщик рассчитывает признаки, detach обрывает граф; AdamW получает только новую Linear-голову с 34 числами. Она учится на 60 новых метках, оценка идёт по 100 отложенным строкам. Проверяется нулевое изменение обоих параметров кодировщика.', 'Точность новой головы выше 0.85; веса и смещение кодировщика остаются неизменными.', exercise='Перед let encoder=source.hidden сохрани let pretrained=source.hidden.clone(); и создай let new_head=LinearConfig::new(16,2).init(&device);. Замени инициализацию head на new_head.clone(), чтобы обе ветви начинали с одинаковой головы. Замени labels на x.clone().powf_scalar(2.0).sum_dim(1).lower_elem(1.0).int().reshape([400]); в исходном выражении labels заранее используй x.clone(), сохранив координаты. Для второй ветви создай let tuned=Net{hidden:pretrained,output:new_head}; передай его в fit(tuned,x.clone().slice([0..60]),labels.clone().slice([0..60])). fit вызывает forward по исходным x без detach и обновляет оба слоя. Замороженную ветвь оставь прежней и сохрани для неё assert_eq!(drift,0.0); для tuned сравни изменение hidden.weight с old_weight.clone() (сохрани клон до расчёта drift) и выведи его, не требуя нулевого изменения. Сравни обе точности по тем же x[300..400] и labels[300..400], сохранив клоны тензоров перед потребляющими вызовами. Порог assert!(accuracy>0.85) был для прямой: в опыте с кругом замени его выводом обеих измеренных точностей.')

    add('meta', r'''
use burn::{module::Param,nn::{Linear,LinearConfig},optim::{AdamWConfig,GradientsParams}};
fn adapt(mut model:Linear,x:Tensor<2>,y:Tensor<2>)->Linear {
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.0).init();
    for _ in 0..10 {
        let loss=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.02,model,grads);
    }
    model
}
fn main(){
    let device=device().autodiff(); let mut initial=LinearConfig::new(1,1).init(&device);
    for _ in 0..120 {
        let x=Tensor::<2>::random([12,1],Distribution::Normal(0.0,1.0),&device);
        let slope=Tensor::<2>::random([1,1],Distribution::Uniform(1.0,2.0),&device);
        let bias=Tensor::<2>::random([1,1],Distribution::Uniform(0.0,0.4),&device);
        let task=adapt(initial.clone(),x.clone(),(x*slope+bias).detach());
        initial.weight=Param::from_tensor((initial.weight.val()*0.85+task.weight.val()*0.15).detach());
        initial.bias=Some(Param::from_tensor((initial.bias.as_ref().unwrap().val()*0.85+task.bias.as_ref().unwrap().val()*0.15).detach()));
    }
    let x=Tensor::<2>::random([30,1],Distribution::Normal(0.0,1.0),&device); let y=x.clone()*1.7+0.15;
    let before=(initial.forward(x.clone().slice([5..30]))-y.clone().slice([5..30])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let adapted=adapt(initial,x.clone().slice([0..5]),y.clone().slice([0..5]));
    let after=(adapted.forward(x.slice([5..30]))-y.slice([5..30])).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Reptile: до {before:.6}, после 10 шагов {after:.6}");
    assert!(after<before && after<0.02);
}
''', 'На каждой из 120 задач клонируется Linear, выполняются 10 шагов AdamW, затем начальные параметры сдвигаются на 15% к адаптированным. detach между задачами исключает градиент через внутреннюю оптимизацию: это Reptile. Для новой прямой только пять примеров участвуют в адаптации; остальные 25 измеряют MSE.', 'Десять шагов по пяти примерам уменьшают MSE на 25 отложенных точках до уровня ниже 0.02.', limits='Один пример адаптации внутри семейства прямых. Он не доказывает преимущество перед обычной инициализацией на множестве задач.', exercise='Создай контрольную LinearConfig::new(1,1).init(&device), передай adapt те же пять x и y и сравни обе MSE. Повтори для 20 новых slope и bias.')

    add('icl', network()+r'''
fn tasks(n:usize,device:&Device)->(Tensor<2>,Tensor<2>){
    let a=Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device);
    let b=Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device);
    let query=Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device);
    let one=Tensor::<2>::ones([n,1],device);
    (Tensor::cat(vec![-one.clone(),-a.clone()+b.clone(),one,a.clone()+b.clone(),query.clone()],1),(a*query+b).detach())
}
fn main(){
    let device=device().autodiff(); let mut model=Net::new(5,64,1,&device);
    let (probe,target)=tasks(100,&device);
    let before=(model.forward(probe.clone())-target.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let mut optimizer=AdamWConfig::new().with_weight_decay(0.01).init();
    for _ in 0..500 {
        let (x,y)=tasks(128,&device); let loss=(model.forward(x)-y).powf_scalar(2.0).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&model);model=optimizer.step(0.01,model,grads);
    }
    // Новые правила после обучения; никаких шагов optimizer на тесте.
    let (x,y)=tasks(100,&device);
    let error=(model.forward(x.clone())-y.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let no_context=x.slice_assign([0..100,1..2],Tensor::<2>::zeros([100,1],&device)).slice_assign([0..100,3..4],Tensor::<2>::zeros([100,1],&device));
    let ablated=(model.forward(no_context)-y).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("До обучения {before:.5}; новые правила {error:.5}; ответы контекста обнулены {ablated:.5}");
    assert!(error<0.02 && error<before*0.1 && ablated>error*5.0);
}
''', 'Net учится на 500 пакетах по 128 правил y=ax+b. В пяти входах закодированы две пары и новый запрос. После обучения 100 новых правил проверяются без optimizer.step; обнуление ответов контекста проверяет, что сеть действительно использует примеры.', 'MSE новых правил меньше 0.02, а обнуление ответов контекста увеличивает ошибку более чем в пять раз.', exercise='В tasks измени x1=-1 и x2=1 на случайные Tensor::<2>::random([n,1],Distribution::Uniform(-1.0,1.0),device); передавай их значения и a*x+b в Tensor::cat. Сравни ошибку, особенно для близких x1 и x2.')

    add('preferences', r'''
use burn::{module::{Module,Param},optim::{AdamConfig,GradientsParams},tensor::activation::{log_softmax,softmax,log_sigmoid}};
#[derive(Module,Debug)]
struct Policy { logits:Param<Tensor<1>> }
fn main(){
    let device=device().autodiff();
    let mut policy=Policy{logits:Param::from_tensor(Tensor::<1>::zeros([2],&device))};
    let reference=log_softmax(Tensor::<1>::zeros([2],&device),0).detach();
    let reference_before=reference.clone(); let mut optimizer=AdamConfig::new().init();
    let before=softmax(policy.logits.val(),0).slice([1..2]).into_scalar::<f32>();
    for _ in 0..100 {
        let logp=log_softmax(policy.logits.val(),0);
        let advantage=(logp.clone().slice([1..2])-logp.slice([0..1]))-(reference.clone().slice([1..2])-reference.clone().slice([0..1]));
        let loss=-log_sigmoid(advantage*0.5).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&policy);policy=optimizer.step(0.05,policy,grads);
    }
    let probabilities=softmax(policy.logits.val(),0);
    let preferred=probabilities.clone().slice([1..2]).into_scalar::<f32>();
    println!("Вероятности {:?}; раньше ответ 1: {before}",probabilities.into_data().try_to_vec::<f32>().unwrap());
    assert!(preferred>0.95 && preferred>before);
    assert_eq!((reference-reference_before).abs().max().into_scalar::<f32>(),0.0);
}
''', 'Param хранит два обучаемых логита. DPO вычитает разность log_softmax неизменной reference из предпочтения политики; log_sigmoid с коэффициентом 0.5 задаёт ошибку. Adam выполняет 100 реальных градиентных обновлений.', 'Вероятность предпочитаемого ответа 1 превышает 0.95; исходная reference не меняется.', exercise='Добавь loss_reverse=-log_sigmoid(-advantage.clone()*0.5).mean(); для этого клонируй advantage в прямом loss. Усредни обе ошибки. Замени assert!(preferred>0.95 && preferred>before) на assert!((preferred-0.5).abs()<0.01), поскольку равные противоречивые оценки оставляют вероятности около 0.5. Проверку неизменности reference сохрани.')
