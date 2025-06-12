
use rand::Rng;
//use std::ops::{AddAssign,MulAssign};
extern crate fasthash;
use ndarray::{Array2,Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

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
    //for i in 0..20 {
    //    println!("{}", hash_with_inputs(0,1,i));
    //}
    /*
    for i in 0..20{
        let mut input: Array2<f64> = Array2::zeros((10,10));
        populate_matrix(&mut input, i);
        //println!("{:?}", input);
        println!("{}", input.sum());
    }
    // */
    let m = 10;
    let n = 10;
    let input = Array2::random((m, n), Uniform::new(0., 1.));
    //println!("{:8.4}", a);
    let sparse_input : CsMat<f64> = CsMat::csc_from_dense(input.view(), 0.);
    let col: usize = 0;

    /*
    let bleh = sparse_input.indptr().outer_inds(1);
    println!("{:?}", sparse_input);
    for i in bleh {
        println!("{:?}", sparse_input.data()[i]);
    }
    */
    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;
    let jl_dim = ((n as f64).log2() *jl_factor).ceil() as usize;

    let sparse_result1 = jl_sketch_sparse(&sparse_input, jl_factor, seed);
    let result2 = jl_sketch_naive(&input, jl_factor, seed);
    let sparse_result2 : CsMat<f64> = CsMat::csc_from_dense(result2.view(), -10.);
    assert_eq!(sparse_result1, sparse_result2);

    let mut blocked_result: CsMat<f64> = CsMat::zero((n,jl_dim));
    jl_sketch_sparse_blocked(&sparse_input, &mut blocked_result, jl_dim, seed, 3, 3);
    assert_eq!(sparse_result1, blocked_result);



    //try_row_populate(20);
    
    }