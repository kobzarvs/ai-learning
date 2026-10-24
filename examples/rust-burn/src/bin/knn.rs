use burn::tensor::{Device, Tensor};
fn main() {
    let device = Device::flex();
    let objects = Tensor::<2>::from_data([[0.0_f32,0.],[1.,1.],[3.,3.]], &device);
    let query = Tensor::<2>::from_data([[0.9_f32,1.1]], &device);
    let distances = (objects-query).powf_scalar(2.0).sum_dim(1);
    let values = distances.into_data().try_to_vec::<f32>().unwrap();
    let nearest = values.iter().enumerate().min_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
    println!("Квадраты расстояний: {values:?}; ближайший: {nearest}");
    assert_eq!(nearest, 1);
}
