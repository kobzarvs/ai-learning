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
