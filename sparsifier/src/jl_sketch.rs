use ndarray::{Array1,Array2,arr1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use fasthash::xx::Hasher64 as XXHasher;
use crate::fasthash::FastHasher;
use std::hash::{Hash,Hasher};



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


pub fn populate_matrix(input: &mut Array2<i64>, seed: u64) {
    let rows = input.dim().0;
    let cols = input.dim().1;
    for i in 0..rows {
        for j in 0..cols {
            input[[i,j]] += hash_with_inputs(seed, i as u64, j as u64);
        }
    }

}
