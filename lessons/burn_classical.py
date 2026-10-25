"""Classical learning tasks; registered by the central Burn catalog."""

# Burn generates actual samples; discrete control-flow algorithms stay in Rust.
NORMAL = r'''
fn normal(n: usize, device: &Device) -> Vec<f32> {
    Tensor::<1>::random([n], Distribution::Normal(0.0, 1.0), device)
        .into_data().try_to_vec::<f32>().unwrap()
}
'''
UNIFORM = r'''
fn uniform(n: usize, device: &Device) -> Vec<f32> {
    Tensor::<1>::random([n], Distribution::Uniform(0.0, 1.0), device)
        .into_data().try_to_vec::<f32>().unwrap()
}
'''


def register(add):
    add('trees', NORMAL + r'''
fn main() {
    let device = device();
    let samples = normal(480, &device);
    let x: Vec<[f32; 2]> = samples.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let y: Vec<bool> = x.iter().map(|p| p[0]+0.5*p[1]>0.0).collect();
    let mut best = (usize::MAX, 0, 0.0, false, false);
    for feature in 0..2 {
        for step in 0..30 {
            let threshold = -2.0+4.0*step as f32/29.0;
            let mut count = [0_usize; 2]; let mut positive = [0_usize; 2];
            for i in 0..180 {
                let side = usize::from(x[i][feature]>=threshold);
                count[side]+=1; positive[side]+=usize::from(y[i]);
            }
            if count.contains(&0) { continue; }
            let a = positive[0]*2>=count[0]; let b = positive[1]*2>=count[1];
            let error = (0..180).filter(|&i| (if x[i][feature]<threshold {a} else {b}) != y[i]).count();
            if error<best.0 { best=(error, feature, threshold, a, b); }
        }
    }
    let (error, feature, threshold, a, b)=best;
    let accuracy=(180..240).filter(|&i| (if x[i][feature]<threshold {a} else {b})==y[i]).count() as f32/60.0;
    println!("Признак {feature}, порог {threshold}, учебных ошибок {error}; новые строки: {accuracy}");
    assert!(error<50 && accuracy>0.7);
}
''', 'Перебираем обе координаты и 30 порогов. Лист предсказывает большинство только первых 180 строк; последние 60 оценивают правило. Burn создаёт нормальные точки, перебор дискретных правил выполняет Rust.',
        'Одно разбиение даёт более 70% правильных ответов на 60 новых точках.',
        exercise='В Rust выдели подбор разбиения в функцию и вызови её отдельно для каждой половины учебной выборки. Запрети листья меньше 10 строк.')

    add('svm', r'''
use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams}};
fn main() {
    let device = device().autodiff();
    let x=Tensor::<2>::random([240,2], Distribution::Normal(0.0,1.0), &device);
    let labels=(x.clone().slice([0..240,0..1])+x.clone().slice([0..240,1..2])*0.5)
        .greater_elem(0.0).float();
    let y=labels.clone()*2.0-1.0;
    let mut model=LinearConfig::new(2,1).init(&device);
    let mut optimizer=AdamConfig::new().init();
    let objective = |score: Tensor<2>, weight: Tensor<2>| {
        (y.clone().slice([0..180])*score*(-1.0)+1.0).clamp_min(0.0).mean()
            +weight.powf_scalar(2.0).sum()*0.01
    };
    let before=objective(model.forward(x.clone().slice([0..180])),model.weight.val()).into_scalar::<f32>();
    for _ in 0..250 {
        let loss=objective(model.forward(x.clone().slice([0..180])),model.weight.val());
        let grads=GradientsParams::from_grads(loss.backward(), &model);
        model=optimizer.step(0.03,model,grads);
    }
    let after=objective(model.forward(x.clone().slice([0..180])),model.weight.val()).into_scalar::<f32>();
    let accuracy=model.forward(x.slice([180..240])).greater_elem(0.0)
        .equal(labels.slice([180..240]).greater_elem(0.5)).float().mean().into_scalar::<f32>();
    println!("Hinge + L2: {before} → {after}; новые строки: {accuracy}");
    assert!(after<before && after<0.45 && accuracy>0.9);
}
''', 'Burn обучает линейный слой градиентами hinge loss с L2-штрафом 0.01. Знаки ответов равны −1/+1. Adam выполняет 250 шагов на 180 строках, затем отдельно проверяются 60 новых.',
        'Hinge + L2 уменьшается, точность новых точек превышает 90%.',
        limits='Линейный мягкий зазор. Используется Adam с явным L2, вместо AdamW Python; ядра и нелинейные границы отсутствуют.',
        exercise='Увеличь множитель 0.01 до 0.2. Выведи сумму квадратов model.weight и сравни точность и размер весов.')

    add('naivebayes', NORMAL + r'''
fn main() {
    let device=device(); let noise=normal(480,&device);
    // Чередование классов даёт перемешанную выборку с обоими классами в каждом срезе.
    let x: Vec<[f32;2]>=(0..240).map(|i| {let shift=if i%2==0 {-1.0} else {1.0}; [noise[2*i]+shift,noise[2*i+1]+shift]}).collect();
    let mut means=[[0.0_f32;2];2]; let mut variance=[[0.0_f32;2];2]; let mut counts=[0_usize;2];
    for i in 0..180 { let k=i%2; counts[k]+=1; for j in 0..2 {means[k][j]+=x[i][j];} }
    for k in 0..2 {for j in 0..2 {means[k][j]/=counts[k] as f32;}}
    for i in 0..180 {for j in 0..2 {variance[i%2][j]+=(x[i][j]-means[i%2][j]).powi(2);}}
    for k in 0..2 {for j in 0..2 {variance[k][j]=variance[k][j]/(counts[k]-1) as f32+1e-4;}}
    // Burn вычисляет логарифмы плотностей одновременно для 60 × 2 × 2 значений.
    let test=Tensor::<2>::from_data(burn::tensor::TensorData::new(x[180..].iter().flatten().copied().collect::<Vec<_>>(),[60,2]),&device).unsqueeze_dim::<3>(1);
    let mu=Tensor::<2>::from_data(means,&device).unsqueeze_dim::<3>(0);
    let var=Tensor::<2>::from_data(variance,&device).unsqueeze_dim::<3>(0);
    let prior=Tensor::<1>::from_data([counts[0] as f32/180.0,counts[1] as f32/180.0],&device).log().unsqueeze_dim::<2>(0);
    let score=(((test-mu).powf_scalar(2.0)/var.clone()+var.log()).sum_dim(2).squeeze_dim::<2>(2)*(-0.5))+prior;
    let predicted=score.argmax(1).into_data().try_to_vec::<i32>().unwrap();
    let accuracy=predicted.iter().enumerate().filter(|&(i,k)| *k as usize==(180+i)%2).count() as f32/60.0;
    println!("Средние: {means:?}; точность: {accuracy}"); assert!(accuracy>0.8);
}
''', 'Оценки средних, несмещённых дисперсий и долей используют только 180 учебных строк. Burn вычисляет гауссовы логарифмические оценки и argmax для 60 новых строк; константа 2π одинакова для классов и сокращается.',
        'Гауссов наивный Байес правильно классифицирует более 80% новых объектов.',
        exercise='Добавь третью координату — копию первой с малым шумом. Расширь массивы и тензоры с 2 до 3 признаков и сравни разность логарифмических оценок классов.')

    add('cluster unsupervised', NORMAL + r'''
fn main() {
    let device=device(); let noise=normal(400,&device);
    let x: Vec<[f32;2]>=(0..200).map(|i| {let shift=if i<100 {-1.0} else {1.0}; [noise[2*i]*0.3+shift,noise[2*i+1]*0.3+shift]}).collect();
    let mut centers=[x[7],x[137]]; let mut labels=vec![0;200]; let mut first=0.0;
    let data=Tensor::<2>::from_data(burn::tensor::TensorData::new(x.iter().flatten().copied().collect::<Vec<_>>(),[200,2]),&device);
    for iteration in 0..20 {
        let c=Tensor::<2>::from_data(centers,&device);
        let distance=(data.clone().unsqueeze_dim::<3>(1)-c.unsqueeze_dim::<3>(0)).powf_scalar(2.0).sum_dim(2).squeeze_dim::<2>(2);
        if iteration==0 { first=distance.clone().min_dim(1).mean().into_scalar::<f32>(); }
        labels=distance.argmin(1).into_data().try_to_vec::<i32>().unwrap().iter().map(|&k| k as usize).collect();
        let mut sum=[[0.0;2];2]; let mut count=[0;2];
        for (p,&k) in x.iter().zip(&labels) {count[k]+=1; for j in 0..2 {sum[k][j]+=p[j];}}
        for k in 0..2 {if count[k]>0 {for j in 0..2 {centers[k][j]=sum[k][j]/count[k] as f32;}}}
    }
    let error=x.iter().zip(&labels).map(|(p,&k)| (p[0]-centers[k][0]).powi(2)+(p[1]-centers[k][1]).powi(2)).sum::<f32>()/200.0;
    let count=labels.iter().filter(|&&k| k==0).count();
    println!("Центры {centers:?}; размеры [{count}, {}]; ошибка {first} → {error}",200-count);
    assert!(error<=first+1e-5 && error<0.3 && (90..=110).contains(&count));
    assert!(centers.iter().all(|c| (c[0].abs()-1.0).abs()<0.15 && (c[1].abs()-1.0).abs()<0.15));
    assert!(centers[0][0]*centers[1][0]<0.0);
}
''', 'Burn рассчитывает матрицу квадратов расстояний и ближайшие центры; Rust обновляет средние непустых групп. Двадцать шагов k-средних не используют метки исходных облаков. Начальные центры — две разные строки данных.',
        'Два центра около [−1,−1] и [1,1], около 100 точек в группе; внутригрупповая ошибка уменьшается.',
        exercise='Задай centers=[x[7],x[137],x[40]], sum=[[0.0;2];3], count=[0;3] и замени цикл обновления for k in 0..2 на 0..3. После обучения посчитай три размера: let counts: Vec<usize>=(0..3).map(|k| labels.iter().filter(|&&v| v==k).count()).collect(); выведи их. Последние три assert, рассчитанные на два центра, замени на assert!(error<=first+1e-5 && error<0.3); assert_eq!(counts.iter().sum::<usize>(),200); assert!(counts.iter().all(|&n| n>0));. Сравни, какое облако разделилось и насколько уменьшилась ошибка.')

    add('decomposition', r'''
use burn::linalg::svd;
fn main() {
    let device=device();
    let t=Tensor::<2>::from_data(burn::tensor::TensorData::new((0..120).map(|i| -2.0+4.0*i as f32/119.0).collect::<Vec<_>>(),[120,1]),&device);
    let x=t.matmul(Tensor::<2>::from_data([[1.0_f32,2.0,-1.0]],&device))
        +Tensor::<2>::random([120,3],Distribution::Normal(0.0,0.05),&device);
    let train=x.clone().slice([0..90]); let mean=train.clone().mean_dim(0);
    let (_,s,vt)=svd::<2,1>(train-mean.clone(),100);
    let direction=vt.slice([0..1,0..3]); let test=x.slice([90..120]);
    let compressed=(test.clone()-mean.clone()).matmul(direction.clone().transpose());
    assert_eq!(compressed.dims(),[30,1]);
    let restored=compressed.matmul(direction.clone())+mean;
    let error=(restored-test).powf_scalar(2.0).mean().into_scalar::<f32>();
    let unit=direction.powf_scalar(2.0).sum().into_scalar::<f32>();
    println!("[30,3] → [30,1]; MSE {error}; сингулярные числа {:?}",s.into_data());
    assert!(error<0.01 && (unit-1.0).abs()<1e-3);
}
''', 'Burn центрирует только первые 90 строк и выполняет SVD. Первая правая сингулярная компонента сжимает последние 30 трёхмерных точек до одного числа и восстанавливает их.',
        'Форма меняется с [30,3] на [30,1], ошибка восстановления ниже 0.01.',
        exercise='Создай дополнительный Tensor случайного шума формы [120,1] с σ=1 и добавь его ко второму столбцу перед SVD. Выведи direction и сравни ошибку.')

    add('isolation', NORMAL + UNIFORM + r'''
fn depth(x: &[[f32;2]], target: usize, draws: &[f32], cursor: &mut usize) -> usize {
    let mut active: Vec<usize>=(0..x.len()).collect();
    for level in 0..30 {
        if active.len()<=1 {return level;}
        let feature=usize::from(draws[*cursor]>=0.5); *cursor+=1;
        let low=active.iter().map(|&i| x[i][feature]).fold(f32::INFINITY,f32::min);
        let high=active.iter().map(|&i| x[i][feature]).fold(f32::NEG_INFINITY,f32::max);
        if high==low {return level;}
        let cut=low+draws[*cursor]*(high-low); *cursor+=1;
        let side=x[target][feature]<cut;
        active.retain(|&i| (x[i][feature]<cut)==side);
        assert!(active.contains(&target));
    } 30
}
fn main() {
    let device=device(); let noise=normal(200,&device);
    let mut x: Vec<[f32;2]>=noise.chunks_exact(2).map(|p| [p[0]*0.3,p[1]*0.3]).collect(); x.push([5.0,5.0]);
    let draws=uniform(12000,&device); let mut cursor=0;
    let ordinary=(0..100).map(|_| depth(&x,0,&draws,&mut cursor) as f32).sum::<f32>()/100.0;
    let outlier=(0..100).map(|_| depth(&x,100,&draws,&mut cursor) as f32).sum::<f32>()/100.0;
    println!("Средняя глубина: обычная {ordinary}; далёкая {outlier}");
    assert!(outlier<ordinary && outlier<4.0 && ordinary>4.0);
}
''', 'Burn генерирует облако и случайные координаты/пороги. Rust сохраняет исходный индекс цели при последовательном фильтровании точек. Для каждой цели усредняются 100 независимых случайных путей.',
        'Точка [5,5] изолируется заметно меньшим числом разрезов.',
        exercise='Добавь 20 точек с небольшим шумом вокруг [5,5]. Увеличь число случайных draws и сравни глубины обычной точки и одной точки новой группы.')

    add('timeseries classicaltime', r'''
use burn::{nn::LinearConfig, optim::{AdamConfig, GradientsParams}};
fn main() {
    let device=device().autodiff();
    let time=Tensor::<1>::from_data(burn::tensor::TensorData::new((0..300).map(|i| i as f32*0.1).collect::<Vec<_>>(),[300]),&device);
    let signal=time.sin()+Tensor::<1>::random([300],Distribution::Normal(0.0,0.05),&device);
    let columns: Vec<Tensor<2>>=(0..6).map(|offset| signal.clone().slice([offset..offset+294]).unsqueeze_dim::<2>(1)).collect();
    let x=Tensor::cat(columns,1); let y=signal.slice([6..300]).unsqueeze_dim::<2>(1);
    let mut model=LinearConfig::new(6,1).init(&device); let mut optimizer=AdamConfig::new().init();
    let initial=(model.forward(x.clone().slice([220..294]))-y.clone().slice([220..294])).powf_scalar(2.0).mean().into_scalar::<f32>();
    for _ in 0..200 {
        let loss=(model.forward(x.clone().slice([0..220]))-y.clone().slice([0..220])).powf_scalar(2.0).mean();
        let grads=GradientsParams::from_grads(loss.backward(),&model); model=optimizer.step(0.02,model,grads);
    }
    let error=(model.forward(x.clone().slice([220..294]))-y.clone().slice([220..294])).powf_scalar(2.0).mean().into_scalar::<f32>();
    let baseline=(x.slice([220..294,5..6])-y.slice([220..294])).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("Ошибка до {initial}; после {error}; повтор последнего {baseline}");
    assert!(error<initial && error<0.02);
    // Сравнение с последним значением измеряется, а не гарантируется любым шумом/стартом.
}
''', 'Burn собирает 294 окна из шести отсчётов и следующего ответа. Линейный слой обучается Adam на первых 220 окнах, затем сравнивается с повторением последнего отсчёта на 74 более поздних окнах.',
        'Ошибка обученного прогноза меньше начальной и ниже 0.02; рядом выводится ошибка повторения последнего значения.',
        limits='Окна сохраняют порядок времени и используют настоящую прошлую историю. Adam вместо AdamW с нулевым decay в Python. Превосходство над повторением последнего зависит от шума и инициализации.',
        exercise='В создании time измени наклон после индекса 220, сохранив непрерывную фазу. Сравни error и baseline на поздних окнах.')

    add('bayes', r'''
fn main() {
    let device=device();
    let observations=[1_usize,1,0,1,1,0,1,1];
    let a=1+observations.iter().sum::<usize>();
    let b=1+observations.len()-observations.iter().sum::<usize>();
    // Для целого shape Gamma(k,1) — сумма k независимых Exp(1).
    // X/(X+Y), X~Gamma(a,1), Y~Gamma(b,1), имеет Beta(a,b).
    let gamma_a=Tensor::<2>::random([10000,a],Distribution::Uniform(0.0,1.0),&device)
        .clamp_min(1e-7).log().sum_dim(1)*(-1.0);
    let gamma_b=Tensor::<2>::random([10000,b],Distribution::Uniform(0.0,1.0),&device)
        .clamp_min(1e-7).log().sum_dim(1)*(-1.0);
    let samples=gamma_a.clone()/(gamma_a+gamma_b);
    let empirical=samples.clone().mean().into_scalar::<f32>();
    let mut values=samples.into_data().try_to_vec::<f32>().unwrap(); values.sort_by(f32::total_cmp);
    let lower=values[250]; let upper=values[9749]; let mean=a as f32/(a+b) as f32;
    println!("Beta({a},{b}); среднее {mean}; выборочное {empirical}; 95% [{lower}, {upper}]");
    assert_eq!((a,b),(7,3)); assert!((mean-0.7).abs()<1e-6);
    assert!((empirical-mean).abs()<0.015 && lower>0.35 && lower<0.5 && upper>0.9 && upper<0.98);
}
''', 'После шести успехов и двух неудач параметры равны 7 и 3. Burn выбирает Beta через отношение двух Gamma: целые формы позволяют точно построить Gamma суммами экспоненциальных величин. Rust сортирует 10 000 образцов для интервала.',
        'Среднее 0.7, выборочное среднее рядом; 95% интервал примерно [0.40,0.93].',
        limits='Построение Gamma через суммы экспонент в этом примере требует целых параметров. Интервал относится к независимым испытаниям с постоянной вероятностью.',
        exercise='Замени observations на Vec из 800 значений с 600 успехами и 200 неудачами. Вместо проверки (7,3) проверь (601,201); сравни ширину интервала, адаптировав границы assert.')

    add('gp', r'''
use burn::linalg::svd;
fn kernel(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    ((a.unsqueeze_dim::<3>(1)-b.unsqueeze_dim::<3>(0)).powf_scalar(2.0)
        .sum_dim(2).squeeze_dim::<2>(2)*(-0.5)).exp()
}
fn main() {
    let device=device();
    let x=Tensor::<2>::from_data([[-1.0_f32],[0.0],[1.0]],&device); let y=x.clone().sin();
    let query=Tensor::<2>::from_data(burn::tensor::TensorData::new((0..21).map(|i| -3.0+i as f32*0.3).collect::<Vec<_>>(),[21,1]),&device);
    let k=kernel(x.clone(),x.clone())+Tensor::<2>::from_data([[0.01_f32,0.0,0.0],[0.0,0.01,0.0],[0.0,0.0,0.01]],&device);
    let cross=kernel(query,x);
    // SVD решает K z = rhs без замены обучения/ядра. Здесь K положительно определена.
    let (u,s,vt)=svd::<2,1>(k.clone(),100);
    let inverse=(vt.transpose()*s.recip().unsqueeze_dim::<2>(0)).matmul(u.transpose());
    let alpha=inverse.clone().matmul(y.clone());
    let residual=(k.matmul(alpha.clone())-y).abs().max().into_scalar::<f32>();
    let mean=cross.clone().matmul(alpha);
    let variance=(cross.clone()*inverse.matmul(cross.transpose()).transpose()).sum_dim(1)*(-1.0)+1.0;
    let variance=variance.clamp_min(0.0).into_data().try_to_vec::<f32>().unwrap();
    let means=mean.into_data().try_to_vec::<f32>().unwrap();
    for i in [0,10,20] {println!("x={}, среднее {}, σ={}",-3.0+i as f32*0.3,means[i],variance[i].sqrt());}
    assert!(residual<1e-4 && variance[10]<0.02 && variance[0]>0.8 && variance[20]>0.8);
    assert!(means[10].abs()<1e-4 && (means[0]+means[20]).abs()<1e-3);
}
''', 'Burn строит RBF-ядро для трёх наблюдений и 21 запроса. SVD решает ту же положительно определённую систему, что solve в Python; проверяется невязка. Формула условной дисперсии показывает рост неопределённости вне наблюдений.',
        'В нуле σ меньше 0.15, на краях ±3 σ больше 0.89; система решена с невязкой меньше 1e−4.',
        exercise='В kernel перед powf_scalar раздели разность на 0.3. Сравни средние и σ в промежутках и у краёв; ожидаемые числовые границы при изменении ядра пересмотри.')

    add('states', NORMAL + r'''
fn main() {
    let device=device(); let noise=normal(60,&device);
    let truth: Vec<f32>=(0..60).map(|i| 2.0*i as f32/59.0).collect();
    let observed: Vec<f32>=truth.iter().zip(noise).map(|(&t,n)| t+0.35*n).collect();
    let mut mean=0.0_f32; let mut variance=1.0_f32; let mut estimates=Vec::new();
    for &measurement in &observed {
        variance+=0.01; let predicted_variance=variance;
        let gain=variance/(variance+0.35_f32.powi(2));
        mean+=gain*(measurement-mean); variance*=1.0-gain; estimates.push(mean);
        assert!(gain>0.0 && gain<1.0 && variance>0.0 && variance<predicted_variance);
    }
    let target=Tensor::<1>::from_data(burn::tensor::TensorData::new(truth,[60]),&device);
    let raw=(Tensor::<1>::from_data(burn::tensor::TensorData::new(observed,[60]),&device)-target.clone()).powf_scalar(2.0).mean().into_scalar::<f32>();
    let filtered=(Tensor::<1>::from_data(burn::tensor::TensorData::new(estimates,[60]),&device)-target).powf_scalar(2.0).mean().into_scalar::<f32>();
    println!("MSE измерений {raw}; фильтра {filtered}"); assert!(filtered<raw && filtered<0.08);
}
''', 'Rust последовательно обновляет среднее и дисперсию одномерного фильтра Калмана: процессный шум 0.01 и измерительный 0.35². Burn генерирует шум и измеряет ошибки всех 60 оценок. Каждый шаг проверяет уменьшение неопределённости после измерения.',
        'Ошибка фильтра меньше ошибки исходных измерений и ниже 0.08.',
        exercise='Прибавь 2 к truth после индекса 30. Сравни MSE и несколько оценок вокруг скачка при процессном шуме 0.01 и 0.1.')

    add('mixtures', r'''
fn main() {
    let device=device();
    let x=Tensor::cat(vec![Tensor::<2>::random([120,1],Distribution::Normal(-1.5,0.3),&device),Tensor::<2>::random([120,1],Distribution::Normal(1.5,0.3),&device)],0);
    let mut means=Tensor::<2>::from_data([[-1.0_f32,1.0]],&device);
    let mut variance=Tensor::<2>::ones([1,2],&device); let mut weights=Tensor::<2>::ones([1,2],&device)*0.5;
    let mut previous=f32::NEG_INFINITY; let mut first=0.0;
    for iteration in 0..30 {
        let score=((x.clone()-means.clone()).powf_scalar(2.0)/variance.clone()+variance.clone().log())*(-0.5)+weights.clone().log();
        // Стабильная log-sum-exp для проверки монотонности EM.
        let max=score.clone().max_dim(1);
        let likelihood=((score.clone()-max.clone()).exp().sum_dim(1).log()+max).mean().into_scalar::<f32>();
        if iteration==0 {first=likelihood;}
        assert!(likelihood+1e-4>=previous); previous=likelihood;
        let responsibility=burn::tensor::activation::softmax(score,1);
        let row_error=(responsibility.clone().sum_dim(1)-1.0).abs().max().into_scalar::<f32>(); assert!(row_error<1e-5);
        let count=responsibility.clone().sum_dim(0);
        means=(responsibility.clone()*x.clone()).sum_dim(0)/count.clone();
        variance=(responsibility*(x.clone()-means.clone()).powf_scalar(2.0)).sum_dim(0)/count.clone();
        variance=variance.clamp_min(1e-4); weights=count/240.0;
    }
    let mu=means.into_data().try_to_vec::<f32>().unwrap(); let pi=weights.into_data().try_to_vec::<f32>().unwrap();
    println!("Центры {mu:?}; доли {pi:?}; log L {first} → {previous}");
    assert!((mu[0]+1.5).abs()<0.15 && (mu[1]-1.5).abs()<0.15);
    assert!(pi.iter().all(|&p| (p-0.5).abs()<0.05) && previous>first+0.5);
}
''', 'Burn выполняет 30 полных EM-итераций: softmax логарифмов плотности даёт ответственности, затем обновляются средние, дисперсии и доли. Проверяем нормировку ответственностей и неубывание логарифмического правдоподобия.',
        'Центры близки к −1.5 и +1.5, доли близки к 0.5; правдоподобие растёт.',
        exercise='Задай means=[[0.0,0.0]], сохранив равные веса и дисперсии. После цикла первый assert о разделённых центрах замени на assert!((mu[0]-mu[1]).abs()<1e-6); во втором assert замени только previous>first+0.5 на previous>=first-1e-4, сохранив проверку долей. Проверки нормировки ответственностей и монотонности внутри цикла оставь. Выведи итог и объясни сохранение симметрии: EM не может разделить одинаково инициализированные компоненты.')

    add('causal', r'''
fn summary(flag: Tensor<1>, time: Tensor<1>) -> (f32,f32) {
    let n=flag.dims()[0] as f64;
    let treated=flag.clone().sum(); let control=flag.clone()*(-1.0)+1.0;
    let difference=((flag.clone()*time.clone()).sum()/treated.clone()
        -(control.clone()*time.clone()).sum()/control.sum()).into_scalar::<f32>();
    let centered_flag=flag.clone()-flag.mean(); let centered_time=time.clone()-time.mean();
    let covariance=(centered_flag.clone()*centered_time.clone()).sum()/n;
    let scale=((centered_flag.powf_scalar(2.0).sum()/n)*(centered_time.powf_scalar(2.0).sum()/n)).sqrt();
    ((covariance/scale).into_scalar::<f32>(),difference)
}
fn main() {
    let device=device(); let n=1000;
    let difficulty=Tensor::<1>::random([n],Distribution::Uniform(1.0,5.0),&device);
    let assignment=Tensor::<1>::random([n],Distribution::Uniform(0.0,1.0),&device);
    let noise=Tensor::<1>::random([n],Distribution::Uniform(-1.0,1.0),&device);
    let minutes_saved=4.0;
    let completion_time=|instruction: Tensor<1>| difficulty.clone()*10.0+20.0-instruction*minutes_saved+noise.clone();
    let changes=completion_time(Tensor::<1>::ones([n],&device))-completion_time(Tensor::<1>::zeros([n],&device));
    let intervention=changes.clone().mean().into_scalar::<f32>();
    assert!((changes+minutes_saved).abs().max().into_scalar::<f32>()<1e-5);
    let mut differences=Vec::new();
    for random_assignment in [false,true] {
        let probability=if random_assignment {Tensor::<1>::ones([n],&device)*0.5} else {(difficulty.clone()-1.0)*0.2+0.1};
        let flag=assignment.clone().lower(probability).float();
        let (correlation,difference)=summary(flag.clone(),completion_time(flag)); differences.push(difference);
        println!("Случайное назначение {random_assignment}; корреляция {correlation}; разница групп {difference}; парное вмешательство {intervention}");
    }
    assert!((intervention+minutes_saved as f32).abs()<1e-5);
    assert!(differences[0]>0.0 && (differences[1]+minutes_saved as f32).abs()<2.5);
}
''', 'Burn сохраняет одни и те же сложности, помехи и числа назначения. Сравниваем корреляцию и разницу групп при смещённом и случайном назначении. Парное вмешательство меняет только инструкцию для каждого того же задания; проверяются все 1000 изменений.',
        'При назначении сложным заданиям разница групп положительна; случайное назначение приближает её к −4. Парное изменение всегда −4 минуты.',
        exercise='Измени minutes_saved на 0.0 и сравни обе разницы групп. Проверки парного и случайного эффекта уже используют minutes_saved, поэтому менять их не требуется. Затем увеличь n до 10000 и сравни абсолютное отклонение случайной разницы от −minutes_saved.')
