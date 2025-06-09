use ndarray::{Array2,ArrayView2};

use fasthash::xx::Hasher64 as XXHasher;
use crate::fasthash::FastHasher;
use std::hash::{Hash,Hasher};

//use math::round::ceil;

use sprs::{CsMat, CsMatView};
use std::ops::Mul;


//maps hash function output to {-1,1} evenly
fn transform(input: i64) -> i64 {
    let result = ((input >> 31) * 2 - 1 ) as i64;
    result
}


//this function is used to find the value of a hash function seeded with seed, when given two positional arguments. Used to repeatably compute values in the JL sketch matrix
// by passing the coordinates of the position as the two arguments.
pub fn hash_with_inputs(seed: u64, input1: u64, input2: u64) -> i64 {
   let mut checkhash = XXHasher::with_seed(seed);
    input1.hash(&mut checkhash);
    input2.hash(&mut checkhash);
    let result = checkhash.finish() as u32;
    //println!("{}",result);
    transform(result as i64)
}


pub fn populate_matrix(input: &mut Array2<f64>, seed: u64) {
    let rows = input.dim().0;
    let cols = input.dim().1;
    for i in 0..rows {
        for j in 0..cols {
            input[[i,j]] += hash_with_inputs(seed, i as u64, j as u64) as f64;
        }
    }

}

// this function both takes a dense matrix representation of the original matrix 
// (though we expect this matrix to be sparse) and allocates the entire JL matrix (actually dense, pretty big)
// so it's not scalable. can be used for correctness checking
pub fn jl_sketch_naive(og_matrix: &Array2<f64>, jl_factor: f64, seed: u64) -> Array2<f64>{
    let og_rows = og_matrix.dim().0;
    let og_cols = og_matrix.dim().1;
    let jl_dim = ((og_rows as f64).log2() *jl_factor).ceil() as usize;
    let mut sketch_matrix: Array2<f64> = Array2::zeros((og_cols,jl_dim));
    populate_matrix(&mut sketch_matrix, seed);
    let result = og_matrix.dot(&sketch_matrix);
    println!("{:?}", og_matrix);
    println!("{:?}", sketch_matrix);
    println!("{:?}", result);
    result
}

// we assume that og_matrix is sparse. the sketch matrix is always dense by construction.

 pub fn jl_sketch_sparse(og_matrix: &CsMat<f64>, jl_factor: f64, seed: u64) -> CsMat<f64> {
    let og_rows = og_matrix.rows();
    let og_cols = og_matrix.cols();
    let jl_dim = ((og_rows as f64).log2() *jl_factor).ceil() as usize;
    let mut sketch_matrix: Array2<f64> = Array2::zeros((og_cols,jl_dim));
    populate_matrix(&mut sketch_matrix, seed);
    let csr_sketch_matrix : CsMat<f64> = CsMat::csr_from_dense(sketch_matrix.view(), -1.0); // i'm nervous about using csr_from_dense with negative epsilon, but it seems to work
    let result = og_matrix.mul(&csr_sketch_matrix);
    println!("{:?}", og_matrix);
    println!("{:?}", sketch_matrix);
    println!("{:?}", result);
    result
 }
 


pub fn multiplier(og_matrix: &CsMat<f64>, other: &CsMat<f64>) -> CsMat<f64> {
    return og_matrix * other
}