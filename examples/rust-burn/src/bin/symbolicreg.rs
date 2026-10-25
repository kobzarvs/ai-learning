fn feature(x:f64,kind:usize)->f64 { match kind {0=>x,1=>x*x,_=>x.sin()} }
fn main() {
    let mut errors = Vec::new();
    for kind in 0..3 {
        let (mut sf,mut sy,mut sff,mut sfy) = (0.0,0.0,0.0,0.0);
        for i in 0..70 { let x = -2.0+4.0*i as f64/99.0; let f=feature(x,kind); let y=2.0*x*x+0.5; sf+=f;sy+=y;sff+=f*f;sfy+=f*y; }

        // Решение двухпараметрической задачи наименьших квадратов.
        let a = (70.0*sfy-sf*sy)/(70.0*sff-sf*sf); let b = (sy-a*sf)/70.0;
        let error = (70..100).map(|i| {let x=-2.0+4.0*i as f64/99.0; (a*feature(x,kind)+b-(2.0*x*x+0.5)).powi(2)}).sum::<f64>()/30.0;
        println!("{}: a={a}, b={b}, ошибка={error}",["a*x+b","a*x²+b","a*sin(x)+b"][kind]);
        errors.push(error);
        if kind==1 {assert!((a-2.0).abs()<1e-10 && (b-0.5).abs()<1e-10);}
    }
    assert!(errors[1]<1e-20 && errors[0]>1.0 && errors[2]>1.0);
}
