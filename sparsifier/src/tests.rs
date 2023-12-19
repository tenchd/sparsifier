use rand::Rng;
use crate::sparsifiers::Sparsifier;

#[cfg(test)]
mod tests {
    use super::*;

    //check that 1) all nonzero rows have weight >=1, and all zero rows come after all nonzero rows in the list.
    pub fn check_matrix_state(s: &Sparsifier) {
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
                let num1 = rand::thread_rng().gen_range(0..nodesize-1);
                let num2 = rand::thread_rng().gen_range(num1+1..nodesize);
                initial.insert(num1, num2);
            }

            check_matrix_state(&initial);
        }
    }


}
