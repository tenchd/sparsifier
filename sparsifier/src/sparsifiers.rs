use ndarray::{Array1,arr1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use std::fmt;


fn filterer(value: &f64) -> f64 {
    if value < &1.0 {
        *value
    }
    else {
        1.0
    }
}


#[derive(Clone,Copy,Debug)]
pub struct EdgeRow{
    pub endpoint1: usize,
    pub endpoint2: usize,
    pub weight: f64,
    pub is_occupied: bool,
}

impl EdgeRow{
    pub fn new(endpoint1: usize, endpoint2: usize) -> EdgeRow {
        EdgeRow{
            endpoint1: endpoint1,
            endpoint2: endpoint2,
            weight: 1.,
            is_occupied: true,
        }
    }

    pub fn new_empty() -> EdgeRow {
        EdgeRow{
            endpoint1: 0,
            endpoint2: 0,
            weight: 0.,
            is_occupied: false,
        }
    }

    pub fn set_to_zero(&mut self){
        self.endpoint1 = 0;
        self.endpoint2 = 0;
        self.weight = 0.;
        self.is_occupied = false;
    }

    // pub fn display(self){
    //     println!("{}, {}, {}", self.endpoint1, self.endpoint2, self.weight);
    // }
}

impl fmt::Display for EdgeRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}), {}", self.endpoint1, self.endpoint2, self.weight)
    }
}


// an implementation of the streaming spectral sparsification algorithm described in Section 3.2 of the paper (see readme)
pub struct Sparsifier{
    // number of nodes in the graph
    pub nodesize: usize,
    // approximation error parameter
    pub epsilon: f64,
    // set to be 200 in line 1 of alg pseudocode, probably can be far smaller
    pub beta_constant: usize,
    // set to be 20 in line 3(b) of alg pseudocode, probably can be far smaller
    pub row_constant: usize,
    // parameter defined in line 1 of alg pseudocode
    pub beta: usize, 
    // set to be row_constant*beta*nodesize in line 3(b) of alg pseudocode
    pub max_rows: usize,
    // array of rows we keep explicitly in memory. 
    pub rows: Vec<EdgeRow>,
    // rows array is kept so that all nonzero rows come before all zero rows. this is a pointer to the first zero row.
    pub first_zero_row: usize,
    // if true, print lots of state information whenever sparsifying.
    pub verbose: bool,
}

//second attempt at a sparsifier that works. simpler and more compact design this time

//consider CSR type representation later maybe? check against this version for correctness
//
impl Sparsifier{
    pub fn new(nodesize: usize, epsilon: f64, beta_constant: usize, row_constant: usize, verbose: bool) -> Sparsifier {
        // as per line 1
        let beta = (epsilon.powf(-2.0) * (beta_constant as f64) * (nodesize as f64).log(2.0)).round() as usize;
        // as per 3(b) condition
        let max_rows = nodesize * beta * row_constant;
        // initialize empty matrix
        let rows = vec![EdgeRow::new_empty(); max_rows];
        Sparsifier{
            nodesize: nodesize,
            epsilon: epsilon,
            beta_constant: beta_constant,
            row_constant: row_constant,
            beta: beta,
            max_rows: max_rows,
            rows: rows,
            first_zero_row: 0,
            verbose: verbose,
        }
    }

    pub fn insert(&mut self, v1: usize, v2: usize) {
        // pointer should point at a zero row
        assert!(!self.rows[self.first_zero_row].is_occupied);
        // in the next open row, add ones to the columns corresponding to the endpoints
        self.rows[self.first_zero_row].endpoint1 = v1;
        self.rows[self.first_zero_row].endpoint2 = v2;
        // set weight to 1
        self.rows[self.first_zero_row].weight = 1.;
        // mark the row we just wrote to as occupied
        self.rows[self.first_zero_row].is_occupied = true;
        // increment pointer to next row
        self.first_zero_row += 1;
        // if you've filled all the rows, run the sparsifier
        if self.first_zero_row == self.max_rows {
            self.sparsify();
        }
    }

    pub fn display(&self) {
        //println!("{:?}", &self.rows);
        for row in &self.rows {
            println!("{}", row);
        }
    }

    // currently this is a stub and computes random values for leverage scores. will drop in a better replacement later.
    pub fn estimate_leverage_scores(&self) -> Array1::<f64> {
        let estimates = Array1::random(self.max_rows, Uniform::new(0., 1.));
        //let estimates = vec!([Uniform::New(0., 1.)]; self.max_rows);
        estimates
    }

    pub fn sparsify(& mut self){
        //println!("time to sparsify!");
        //placeholder: random estimates for now
        let estimates = self.estimate_leverage_scores();
        //println!("{:?}\n\n", estimates);
        //multiply all by beta as per 3(b)ii
        let scaled_estimates = estimates * arr1(&[self.beta as f64]);
        //println!("{:?}\n\n", scaled_estimates);
        //estimates less than 1 are squared, those larger than 1 are set to 1. this is a dumb way to do this, fix it later
        let filtered_estimates = scaled_estimates.map(&filterer) * scaled_estimates.map(&filterer);
        //println!("{:?}\n\n", filtered_estimates);
        //subsample each row w/p = its filtered estimate
        let coins = Array1::random(self.max_rows, Uniform::new(0., 1.));
        // if the coin value is less than the filtered estimate, divide the row by sqrt(filtered estimate). else set the row to 0.
        //println!("{:?}\n\n", coins);
        let mut reweight_factors = Array1::<f64>::zeros(self.max_rows);
        for i in 0..=self.max_rows-1 {
            reweight_factors[[i]] += filtered_estimates[[i]].powf(-0.5);
        }
        

        let mut deletions = Vec::new();
        for i in 0..=self.max_rows-1 {
            if coins[[i]] < filtered_estimates[[i]] {
                self.rows[i].weight *= reweight_factors[[i]];

            }
            else {
                self.rows[i].set_to_zero();
                //use deletions as a stack, we'll apply these deletions in reverse order in a minute.
                deletions.push(i);
            }
        }
        
        // for row in &self.rows {
        //     print!("{} ", row.weight);
        // }
        // println!();

        let num_deletions = deletions.len();

        let num_nonzeros = self.max_rows - num_deletions;
        // make all remaining rows zeros
        // since we use deletions as a stack, these deletions should apply in reverse order (and thus delete the correct entries.)
        //see about rewriting using while let Some(top) = deletions.pop()
        while let Some(next_deletion) = deletions.pop() {
            self.rows.remove(next_deletion);
        }
        for _i in num_nonzeros..=self.max_rows-1 {
            self.rows.push(EdgeRow::new_empty());
            //self.rows.row_mut(i).assign(&Array1::<f64>::zeros(self.nodesize));
        }
        // move pointer to beginning of zero rows
        self.first_zero_row = num_nonzeros;


    }

}