
use ndarray::{Array1,Array2,arr1,s};
use rand::Rng;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
//use std::ops::{AddAssign,MulAssign};

fn filterer(value: &f64) -> f64 {
    if value < &1.0 {
        *value
    }
    else {
        1.0
    }
}

pub struct Sparsifier{
    pub nodesize: usize,
    pub epsilon: f64,
    pub beta_constant: usize,
    pub row_constant: usize,
    pub beta: usize, 
    pub max_rows: usize,
    pub rows: Array2<f64>,
    pub first_zero_row: usize,
    pub occupied_rows: Array1<f64>,
}

//test for correcness so far
//consider CSR type representation later maybe? check against this version for correctness
//
impl Sparsifier{
    pub fn new(nodesize: usize, epsilon: f64, beta_constant: usize, row_constant: usize) -> Sparsifier {
        let beta = (epsilon.powf(-2.0) * (beta_constant as f64) * (nodesize as f64).log(2.0)).round() as usize;
        let max_rows = nodesize * beta * row_constant;
        let zeros = Array2::<f64>::zeros((max_rows, nodesize));
        let occupied_rows = Array1::<f64>::zeros(max_rows);
        Sparsifier{
            nodesize: nodesize,
            epsilon: epsilon,
            beta_constant: beta_constant,
            row_constant: row_constant,
            beta: beta,
            max_rows: max_rows,
            rows: zeros,
            first_zero_row: 0,
            occupied_rows: occupied_rows,
        }
    }

    pub fn insert(&mut self, v1: usize, v2: usize) {
        assert!(self.occupied_rows[[self.first_zero_row]]==0.0);
        self.rows[[self.first_zero_row, v1]] += 1.0;
        self.rows[[self.first_zero_row, v2]] += 1.0;
        self.occupied_rows[[self.first_zero_row]] = 1.0;
        self.first_zero_row += 1;
        if self.first_zero_row == self.max_rows {
            self.sparsify();
        }
        //println!("{:?}", self.rows[self.first_zero_row]);
    }

    pub fn display(&self) {
        println!("{:?}", &self.rows);
    }

    pub fn estimate_leverage_scores(&self) -> Array1::<f64> {
        //let ones = Array1::<f64>::ones(self.max_rows);
        //let scaler = arr1(&[0.5]);
        //let estimates = ones * scaler;
        let estimates = Array1::random(self.max_rows, Uniform::new(0., 1.));
        estimates
    }

    pub fn sparsify(& mut self){
        println!("time to sparsify!");
        //placeholder: random estimates for now
        let estimates = self.estimate_leverage_scores();
        //println!("{:?}", estimates);
        //multiply all by beta
        let scaled_estimates = estimates * arr1(&[self.beta as f64]);
        //println!("{:?}", scaled_estimates);
        //estimates less than 1 are squared, those larger than 1 are set to 1.
        let filtered_estimates = scaled_estimates.map(&filterer) * scaled_estimates;
        println!("{:?}", filtered_estimates);
        //subsample each row w/p = its filtered estimate
        let coins = Array1::random(self.max_rows, Uniform::new(0., 1.));
        // if the coin value is less than the filtered estimate, divide the row by sqrt(filtered estimate). else set the row to 0.
        //println!("{:?}", coins);
        let reweight_factors = Array1::<f64>::zeros(self.max_rows);
        for i in 0..=self.max_rows-1 {
            reweight_factors[[i]] += filtered_estimates[[i]].powf(-0.5);
        }
        
        //
        let mut next_row: usize = 0;
        for i in 0..=self.max_rows-1 {
            if coins[[i]] < filtered_estimates[[i]] {
                //let reweight_factor = filtered_estimates[[i]].powf(-0.5);
                self.rows.row_mut(next_row).assign(self.rows.slice(s![0, ..])*arr1(&[reweight_factors[[i]]]));
                next_row +=1;
            }
            else {
                //self.rows.row_mut(i).mul_assign(0.0);
                //self.occupied_rows[[i]] = 0.0;
            }
        }
        let mut num_nonzeros = next_row;
        for i in num_nonzeros..=self.max_rows-1 {
            self.rows.row_mut(i).assign(&Array1::<f64>::zeros(self.nodesize));
        }
        self.first_zero_row = num_nonzeros;


    }

}


fn main() {
    let nodesize: usize  = 10;
    let epsilon: f64 = 1.0;
    let row_constant: usize  = 1;
    let beta_constant: usize = 1;

    let mut initial = Sparsifier::new(nodesize, epsilon, beta_constant, row_constant);

    

    //initial.display();
    for _ in 1..=initial.max_rows {
        let num1 = rand::thread_rng().gen_range(0..nodesize-1);
        let num2 = rand::thread_rng().gen_range(num1+1..nodesize);
        //let (num1, num2) = (rand::thread_rng().gen_range(0..nodesize), rand::thread_rng().gen_range(0..nodesize));
        //let (num1, num2) = (1,2);
        initial.insert(num1, num2);
    }
    //initial.display();
    //initial.insert(1,2);
    initial.display();

    

    //let zeros = Array2::<f64>::zeros((10,10));
    //println!("{:?}", zeros[[0,0]]);
}
