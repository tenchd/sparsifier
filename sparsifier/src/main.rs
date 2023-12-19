
use rand::Rng;
//use std::ops::{AddAssign,MulAssign};
use ndarray::array;


mod sparsifiers;
mod tests;

use sparsifiers::Sparsifier;


fn main() {
    let nodesize: usize  = 10;
    let epsilon: f64 = 1.0;
    let row_constant: usize  = 1;
    let beta_constant: usize = 1;
    let verbose: bool = false;

    let mut initial = Sparsifier::new(nodesize, epsilon, beta_constant, row_constant, verbose);

    

    for _ in 1..=initial.max_rows {
        let num1 = rand::thread_rng().gen_range(0..nodesize-1);
        let num2 = rand::thread_rng().gen_range(num1+1..nodesize);
        //let (num1, num2) = (rand::thread_rng().gen_range(0..nodesize), rand::thread_rng().gen_range(0..nodesize));
        //let (num1, num2) = (1,2);
        initial.insert(num1, num2);
    }

    for _ in 0..10 {
        let num1 = rand::thread_rng().gen_range(0..nodesize-1);
        let num2 = rand::thread_rng().gen_range(num1+1..nodesize);
        //let (num1, num2) = (rand::thread_rng().gen_range(0..nodesize), rand::thread_rng().gen_range(0..nodesize));
        //let (num1, num2) = (1,2);
        initial.insert(num1, num2);
    }


    // let data = array![0.1, 1.32, 4.0, 0.4];
    // let result = data.map(&filterer).powf(2.0);
    // println!("{}", result);

    //initial.display();



    //let zeros = Array2::<f64>::zeros((10,10));
    //println!("{:?}", zeros[[0,0]]);
}
