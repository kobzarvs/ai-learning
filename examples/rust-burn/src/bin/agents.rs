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
