use ndarray::{Array1,Array2};

use fasthash::xx::Hasher64 as XXHasher;
use crate::fasthash::FastHasher;
use std::hash::{Hash,Hasher};
use std::cmp::min;

//use math::round::ceil;

use sprs::{CsMat};
use std::ops::Mul;


//maps hash function output to {-1,1} evenly
fn transform(input: i64) -> i64 {
    let result = ((input >> 31) * 2 - 1 ) as i64;
    result
}


//get all the positions of nonzeros in particular column of sparse matrix representation
/*
pub fn get_nz_indices(input: &Array2<f64>, col: usize) -> {
    let cols = input.dim().1;
    assert!(col <= cols);
    return input.indptr().outer_inds(col)
}
*/

// function to add value 'val' to position 'row', 'col' in sparse matrix. 
pub fn add_to_position(matrix: &mut CsMat<f64>, row: usize, col: usize, val:f64) {
    let location = matrix.get_mut(row,col);
    match location {
        Some(p) => *p += val,
        None => matrix.insert(row, col, val),
    }
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

/* old and busted 
pub fn populate_row(input: &mut Array1<f64>, row: usize, seed: u64){
    let cols = input.dim();
    for col in 0..cols {
        input[[col]] = hash_with_inputs(seed, row as u64, col as u64) as f64;
    }
}
*/

pub fn populate_row(input: &mut Array1<f64>, row: usize, col_start: usize, col_end: usize, seed: u64){
    //let cols = input.dim();
    //let col_end = col_start + cols;
    //println!("{},{}", col_start, col_end);
    let num_cols = col_end - col_start;
    for col in 0..num_cols {
        let actual_col = col+col_start; //have to hash actual column value which should be col+col_start
        input[[col]] = hash_with_inputs(seed, row as u64, actual_col as u64) as f64; 
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
    //println!("{:?}", sketch_matrix);
    let result = og_matrix.dot(&sketch_matrix);
    /*
    println!("{:?}", og_matrix);
    println!("{:?}", sketch_matrix);
    println!("{:?}", result);
    */
    result
}

// this function JL sketches a sparse encoding of the input matrix and outputs in a sparse format as well. 
// it doesn't do blocked operations though, so it's still not scalable because it represents the entire
// dense sketch matrix at all times.
 pub fn jl_sketch_sparse(og_matrix: &CsMat<f64>, jl_factor: f64, seed: u64) -> CsMat<f64> {
    let og_rows = og_matrix.rows();
    let og_cols = og_matrix.cols();
    let jl_dim = ((og_rows as f64).log2() *jl_factor).ceil() as usize;
    let mut sketch_matrix: Array2<f64> = Array2::zeros((og_cols,jl_dim));
    populate_matrix(&mut sketch_matrix, seed);
    let csr_sketch_matrix : CsMat<f64> = CsMat::csr_from_dense(sketch_matrix.view(), -1.0); // i'm nervous about using csr_from_dense with negative epsilon, but it seems to work
    let result = og_matrix.mul(&csr_sketch_matrix);
    /*
    println!("{:?}", og_matrix);
    println!("{:?}", sketch_matrix);
    println!("{:?}", result);
    */
    result
 }
 


pub fn jl_sketch_sparse_blocked(og_matrix: &CsMat<f64>, result_matrix: &mut CsMat<f64>, jl_dim: usize, seed: u64, block_rows: usize, block_cols: usize, display: bool) {
    let og_rows = og_matrix.rows();
    let og_cols = og_matrix.cols();
    //let jl_dim = ((og_cols as f64).log2() *jl_factor).ceil() as usize; //should this be based on rows or cols? I think cols because there are one col for each vertex.
    // make sure you don't set a bigger window size than the JL sketch matrix in either dimension
    let block_row_size = min(og_rows, block_rows);
    let block_col_size = min(jl_dim, block_cols);
    for i in (0..og_cols).step_by(block_row_size){
        for j in (0..jl_dim).step_by(block_col_size){

            // make sure we don't overrun the last column in the JL sketch matrix (can happen when JL output dim isn't a multiple of block_col_size)
            // we're going to iterate from column j to column j+num_cols
            //let num_cols = min(jl_dim + j - 1, block_col_size);
            let inner_cols_max = min(jl_dim, j+block_col_size);
            let inner_cols_min = j;
            let num_cols = inner_cols_max-inner_cols_min;
            // make vector that we'll use to temporarily store a row of the jl sketch matrix.
            let mut jl_temp_row: Array1<f64> = Array1::zeros(num_cols);
            //make sure we don't overrun the last row in the JL sketch matrix
            let inner_rows_min = i;
            let inner_rows_max = min(i+block_row_size, og_cols);

            if display {println!("-----i={},j={},sketch_row_range={}-{},sketch_col_range={}-{}-----", i, j, inner_rows_min, inner_rows_max, inner_cols_min, inner_cols_max);}
            for sketch_row in inner_rows_min..inner_rows_max {
                // grab every nonzero entry in the row_th column of A
                if display {println!("range of nonzeros in column {} of original matrix: {:?}", sketch_row, og_matrix.indptr().outer_inds(sketch_row));}
                for index in og_matrix.indptr().outer_inds(sketch_row) {
                    let nonzero_row_index = og_matrix.indices()[index];
                    if display {println!("{} row index of nonzero in col {} of original matrix: {:?}", index, sketch_row, nonzero_row_index);}
                    populate_row(&mut jl_temp_row, sketch_row, inner_cols_min, inner_cols_max, seed);
                    if display {println!("{:?}", jl_temp_row);}
                    for k in 0..num_cols {
                        if display {println!("sketch_row={},index={},k={}",sketch_row,index,k);}
                        //println!("answer matrix entry {},{} has entry {:?} and we will add {}*{}", sketch_row, j+k, result_matrix.get_mut(j+k, sketch_row), jl_temp_row[[k]], og_matrix.data()[index]);
                        if display {println!("answer matrix entry {},{} has entry {:?} and we will add {}*{}", nonzero_row_index, j+k, result_matrix.get_mut(j+k, sketch_row), jl_temp_row[[k]], og_matrix.data()[index]);}
                        //println!("{:?}", jl_temp_row[[k]]);
                        //println!("{:?}", og_matrix.data()[index]);
                        let new_value: f64 = jl_temp_row[[k]] * og_matrix.data()[index];
                        //add_to_position(result_matrix, j+k, row, new_value);
                        add_to_position(result_matrix, nonzero_row_index, j+k, new_value);
                        
                        //result_matrix.get_mut(j+k, row) += jl_temp_row[[k]] * og_matrix.data()[index];
                    }
                }
            }
               // */
        }
    }


    

}

