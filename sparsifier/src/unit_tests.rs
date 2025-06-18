use rand::Rng;
use crate::sparsifiers::Sparsifier;
use crate::jl_sketch::{populate_row, jl_sketch_sparse_blocked, jl_sketch_sparse};
use ndarray::{Array1,Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform,Bernoulli};
use sprs::{CsMat, CsMatView};

#[cfg(test)]
mod tests {
    use super::*;

    //check that 1) all nonzero rows have weight >=1, and all zero rows come after all nonzero rows in the list.
    fn check_matrix_state(s: &Sparsifier) {
        let mut current_row: usize = 0;
        for row in &s.rows {
            if current_row < s.first_zero_row {
                assert!(row.is_occupied, "row {} is zeroed but first_zero_row is set to {}.", current_row, s.first_zero_row);
                //the below logic doesn't hold if we extend this to handle weighted input edges. then you could have an input edge with weight <1.0 which hasn't been sparsified yet, 
                //or one that is still that small after being sampled and having its weight increased.
                assert!(row.weight >= 1.0, "row {} is marked as nonzero but has weight {} (should be >= 1.0).", current_row, row.weight);
            }
            else {
                assert!(!row.is_occupied, "row {} is nonzero but first_zero_row is set to {}.", current_row, s.first_zero_row);
                assert!(row.weight == 0.0, "row {} is marked as zero but has weight {}.", current_row, row.weight);
            }
            current_row += 1;
        }
    }


    // inserts random edges until it sparsifies, then check matrix state. then add a bunch more edges, inducing more sparsifications, and continue to check periodically.
    #[test]
    fn zero_order() {
        let nodesize: usize  = 10;
        let epsilon: f64 = 1.0;
        let row_constant: usize  = 1;
        let beta_constant: usize = 1;
        let verbose: bool = false;

        let mut initial = Sparsifier::new(nodesize, epsilon, beta_constant, row_constant, verbose);

        
        for _ in 0..10 {
            for _ in 1..=initial.max_rows {
                //let num1 = rand::thread_rng().gen_range(0..nodesize-1);
                //let num2 = rand::thread_rng().gen_range(num1+1..nodesize);
                let mut rng = rand::rng();
                let num1 = rng.random_range(0..nodesize-1);
                let num2 = rng.random_range(num1+1..nodesize);
                initial.insert(num1, num2);
            }

            check_matrix_state(&initial);
        }
    }



    #[test]
    fn try_populate_row() {
        let length = 8;
        let mut row: Array1<f64> = Array1::zeros(length);
        populate_row(&mut row, 0, 0, length, 0);
        for i in 0..length {
            assert!(row[i] != 0.0);
        }
    }

    #[test]
    fn blocked_jlsketch_correct_small() {
        let n = 200; // rows in og matrix = # vertices
        let m = 4000; // cols in og matrix = # edges. this dim will be sketched away

        let block_row_sizes = vec![9,100,500]; // size of block in row dimension
        let block_col_sizes = vec![3,9,20]; // size of block in column dimension

        let seeds = vec![1,2,3,4,5]; // fix random seed for reproducible experiments/tests
        let jl_factors = vec![1.5,4.0]; // factor used in FILL THIS POINTER IN

 
        for i in 0..3 {
            let block_rows = block_row_sizes[i];
            let block_cols = block_col_sizes[i];

            // create random sparse input matrix. make this more efficient later
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
            let sparse_input : CsMat<f64> = CsMat::csc_from_dense(input.view(), 0.);

            for seed in &seeds {
                for jl_factor in &jl_factors {
                    let jl_dim = ((n as f64).log2() *jl_factor).ceil() as usize; // compute new dimension of sketched matrix
                    // sketch using non-blocked method
                    let sparse_result1 = jl_sketch_sparse(&sparse_input, *jl_factor, *seed);
                    //let result2 = jl_sketch_naive(&input, jl_factor, seed);
                    //println!("{:?}", result2);
                    //let sparse_result2 : CsMat<f64> = CsMat::csc_from_dense(result2.view(), -1000000000.);
                    //assert_eq!(sparse_result1, sparse_result2);

                    // sketch using blocked method
                    let mut blocked_result: CsMat<f64> = CsMat::zero((jl_dim,n)).transpose_into();
                    jl_sketch_sparse_blocked(&sparse_input, &mut blocked_result, jl_dim, *seed, block_rows, block_cols, false);
                    
                    //compare answers. they should match
                    assert_eq!(sparse_result1, blocked_result);
                }
            }
        }
    }

    #[test]
    fn blocked_jlsketch_correct_large() {
        let n = 2000; // rows in og matrix = # vertices
        let m = 40000; // cols in og matrix = # edges. this dim will be sketched away

        let block_row_sizes = vec![500]; // size of block in row dimension
        let block_col_sizes = vec![20]; // size of block in column dimension

        let seeds = vec![1]; // fix random seed for reproducible experiments/tests
        let jl_factors = vec![1.5]; // factor used in FILL THIS POINTER IN

 
        for i in 0..1 {
            let block_rows = block_row_sizes[i];
            let block_cols = block_col_sizes[i];

            // create random sparse input matrix. make this more efficient later
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
            let sparse_input : CsMat<f64> = CsMat::csc_from_dense(input.view(), 0.);

            for seed in &seeds {
                for jl_factor in &jl_factors {
                    let jl_dim = ((n as f64).log2() *jl_factor).ceil() as usize; // compute new dimension of sketched matrix
                    // sketch using non-blocked method
                    let sparse_result1 = jl_sketch_sparse(&sparse_input, *jl_factor, *seed);
                    //let result2 = jl_sketch_naive(&input, jl_factor, seed);
                    //println!("{:?}", result2);
                    //let sparse_result2 : CsMat<f64> = CsMat::csc_from_dense(result2.view(), -1000000000.);
                    //assert_eq!(sparse_result1, sparse_result2);

                    // sketch using blocked method
                    let mut blocked_result: CsMat<f64> = CsMat::zero((jl_dim,n)).transpose_into();
                    jl_sketch_sparse_blocked(&sparse_input, &mut blocked_result, jl_dim, *seed, block_rows, block_cols, false);
                    
                    //compare answers. they should match
                    assert_eq!(sparse_result1, blocked_result);
                }
            }
        }
    }


}
