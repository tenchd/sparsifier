
use rand::Rng;
//use std::ops::{AddAssign,MulAssign};
extern crate fasthash;
use ndarray::{Array2,Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform,Bernoulli};

use sprs::{CsMat, CsMatView, CsVec};


mod sparsifiers;
mod unit_tests;
mod jl_sketch;

use sparsifiers::Sparsifier;
use jl_sketch::{hash_with_inputs,populate_matrix,jl_sketch_naive,multiplier,jl_sketch_sparse,try_row_populate,jl_sketch_sparse_blocked};


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
    let n = 200; // rows in og matrix = # vertices
    let m = 4000; // cols in og matrix = # edges. this dim will be sketched away

 // /* 
    let values = Array2::random((n, m), Uniform::new(0., 1.));
    let mask = Array2::random((n, m), Bernoulli::new(0.1).expect("bernoulli gen failed"));
    let mask_closure = |val: &bool| -> f64 {
        if *val {
            return 1.
        }
        return 0.
    };
    let newmask = mask.map(mask_closure);
    let input = &values*&newmask;
 // */
    
  /*
        let mut input: Array2<f64> = Array2::zeros((n,m));
    input[[0,0]] = 0.22;
    input[[9,8]] = 0.91;
    //input[[3,5]] = 0.75;
    //input[[5,2]] = 0.2;

// */

    let sparse_input : CsMat<f64> = CsMat::csc_from_dense(input.view(), 0.);
    //println!("{:?}", input);
    //println!("{:?}", sparse_input);




    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;
    let jl_dim = ((n as f64).log2() *jl_factor).ceil() as usize;
    println!("output matrix is {}x{}", n, jl_dim);

    let sparse_result1 = jl_sketch_sparse(&sparse_input, jl_factor, seed);
    //let result2 = jl_sketch_naive(&input, jl_factor, seed);
    //println!("{:?}", result2);
    //let sparse_result2 : CsMat<f64> = CsMat::csc_from_dense(result2.view(), -1000000000.);
    //assert_eq!(sparse_result1, sparse_result2);

    let mut blocked_result: CsMat<f64> = CsMat::zero((jl_dim,n)).transpose_into();
    let block_rows: usize = 9;
    let block_cols: usize = 9;
    jl_sketch_sparse_blocked(&sparse_input, &mut blocked_result, jl_dim, seed, block_rows, block_cols, false);
    //println!("{:?}",blocked_result);
    //let diff = &blocked_result - &sparse_result2;
    //println!("{:?}", diff);
    assert_eq!(sparse_result1, blocked_result);    
    }

