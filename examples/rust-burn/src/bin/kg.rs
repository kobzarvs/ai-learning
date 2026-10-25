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
