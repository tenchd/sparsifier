
use rand::Rng;
//use std::ops::{AddAssign,MulAssign};
extern crate fasthash;
use ndarray::{Array1,Array2,arr1,array};


mod sparsifiers;
mod unit_tests;
mod jl_sketch;

use sparsifiers::Sparsifier;
use jl_sketch::{hash_with_inputs,populate_matrix};


fn main1() {
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

fn main(){
    //for i in 0..20 {
    //    println!("{}", hash_with_inputs(0,1,i));
    //}
    for i in 0..20{
        let mut input: Array2<i64> = Array2::zeros((10,10));
        populate_matrix(&mut input, i);
        //println!("{:?}", input);
        println!("{}", input.sum());
    }
}