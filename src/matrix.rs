use smallvec::SmallVec;
use std::fmt;

use crate::field::Field;

#[derive(Debug)]
pub enum Error {
    SingularMatrix,
}


macro_rules! acc {
    ($m:ident, $r:expr, $c:expr) => {
        $m.data[$r * $m.col_count + $c]
    };
}

fn flatten<T>(m: Vec<Vec<T>>) -> Vec<T> {
    let mut result: Vec<T> = Vec::with_capacity(m.len() * m[0].len());
    for row in m.into_iter() {
        for v in row.into_iter() {
            result.push(v);
        }
    }
    result
}

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix<F: Field> {
    row_count: usize,
    col_count: usize,
    data: SmallVec<[F::Elem; 1024]>, // store in flattened structure
                                     // the smallvec can hold a matrix of size up to 32x32 in stack
}

fn calc_matrix_row_start_end(col_count: usize, row: usize) -> (usize, usize) {
    let start = row * col_count;
    let end = start + col_count;

    (start, end)
}

impl<F: Field> Matrix<F> {
    fn calc_row_start_end(&self, row: usize) -> (usize, usize) {
        calc_matrix_row_start_end(self.col_count, row)
    }

    pub fn new(rows: usize, cols: usize) -> Matrix<F> {
        let data = SmallVec::from_vec(vec![F::zero(); rows * cols]);

        Matrix {
            row_count: rows,
            col_count: cols,
            data,
        }
    }

    pub fn new_with_data(init_data: Vec<Vec<F::Elem>>) -> Matrix<F> {
        let rows = init_data.len();
        let cols = init_data[0].len();

        for r in init_data.iter() {
            if r.len() != cols {
                panic!("Inconsistent row sizes")
            }
        }

        let data = SmallVec::from_vec(flatten(init_data));

        Matrix {
            row_count: rows,
            col_count: cols,
            data,
        }
    }

    #[cfg(test)]
    pub fn make_random(size: usize) -> Matrix<F>
    where
        rand::distributions::Standard: rand::distributions::Distribution<F::Elem>,
    {
        let mut vec: Vec<Vec<F::Elem>> = vec![vec![Default::default(); size]; size];
        for v in vec.iter_mut() {
            crate::tests::fill_random(v);
        }

        Matrix::new_with_data(vec)
    }

    pub fn identity(size: usize) -> Matrix<F> {
        let mut result = Self::new(size, size);
        for i in 0..size {
            acc!(result, i, i) = F::one();
        }
        result
    }

    pub fn col_count(&self) -> usize {
        self.col_count
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn get(&self, r: usize, c: usize) -> F::Elem {
        acc!(self, r, c).clone()
    }

    pub fn set(&mut self, r: usize, c: usize, val: F::Elem) {
        acc!(self, r, c) = val;
    }

    pub fn multiply(&self, rhs: &Matrix<F>) -> Matrix<F> {
        if self.col_count != rhs.row_count {
            panic!(
                "Colomn count on left is different from row count on right, lhs: {}, rhs: {}",
                self.col_count, rhs.row_count
            )
        }
        let mut result = Self::new(self.row_count, rhs.col_count);
        for r in 0..self.row_count {
            for c in 0..rhs.col_count {
                let mut val = F::zero();
                for i in 0..self.col_count {
                    let mul = F::mul(acc!(self, r, i).clone(), acc!(rhs, i, c).clone());

                    val = F::add(val, mul);
                }
                acc!(result, r, c) = val;
            }
        }
        result
    }

    pub fn augment(&self, rhs: &Matrix<F>) -> Matrix<F> {
        if self.row_count != rhs.row_count {
            panic!(
                "Matrices do not have the same row count, lhs: {}, rhs: {}",
                self.row_count, rhs.row_count
            )
        }
        let mut result = Self::new(self.row_count, self.col_count + rhs.col_count);
        for r in 0..self.row_count {
            for c in 0..self.col_count {
                acc!(result, r, c) = acc!(self, r, c).clone();
            }
            let self_column_count = self.col_count;
            for c in 0..rhs.col_count {
                acc!(result, r, self_column_count + c) = acc!(rhs, r, c).clone();
            }
        }

        result
    }

    pub fn sub_matrix(&self, rmin: usize, cmin: usize, rmax: usize, cmax: usize) -> Matrix<F> {
        let mut result = Self::new(rmax - rmin, cmax - cmin);
        for r in rmin..rmax {
            for c in cmin..cmax {
                acc!(result, r - rmin, c - cmin) = acc!(self, r, c).clone();
            }
        }
        result
    }

    pub fn get_row(&self, row: usize) -> &[F::Elem] {
        let (start, end) = self.calc_row_start_end(row);

        &self.data[start..end]
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        let (r1_s, _) = self.calc_row_start_end(r1);
        let (r2_s, _) = self.calc_row_start_end(r2);

        if r1 == r2 {
            return;
        } else {
            for i in 0..self.col_count {
                self.data.swap(r1_s + i, r2_s + i);
            }
        }
    }

    pub fn is_square(&self) -> bool {
        self.row_count == self.col_count
    }

    pub fn gaussian_elim(&mut self) -> Result<(), Error> {
        for r in 0..self.row_count {
            if acc!(self, r, r) == F::zero() {
                for r_below in r + 1..self.row_count {
                    if acc!(self, r_below, r) != F::zero() {
                        self.swap_rows(r, r_below);
                        break;
                    }
                }
            }
            // If we couldn't find one, the matrix is singular.
            if acc!(self, r, r) == F::zero() {
                return Err(Error::SingularMatrix);
            }
            // Scale to 1.
            if acc!(self, r, r) != F::one() {
                let scale = F::div(F::one(), acc!(self, r, r).clone());
                for c in 0..self.col_count {
                    acc!(self, r, c) = F::mul(scale, acc!(self, r, c).clone());
                }
            }
            // Make everything below the 1 be a 0 by subtracting
            // a multiple of it.  (Subtraction and addition are
            // both exclusive or in the Galois field.)
            for r_below in r + 1..self.row_count {
                if acc!(self, r_below, r) != F::zero() {
                    let scale = acc!(self, r_below, r).clone();
                    for c in 0..self.col_count {
                        acc!(self, r_below, c) = F::add(
                            acc!(self, r_below, c).clone(),
                            F::mul(scale, acc!(self, r, c).clone()),
                        );
                    }
                }
            }
        }

        // Now clear the part above the main diagonal.
        for d in 0..self.row_count {
            for r_above in 0..d {
                if acc!(self, r_above, d) != F::zero() {
                    let scale = acc!(self, r_above, d).clone();
                    for c in 0..self.col_count {
                        acc!(self, r_above, c) = F::add(
                            acc!(self, r_above, c).clone(),
                            F::mul(scale, acc!(self, d, c).clone()),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub fn invert(&self) -> Result<Matrix<F>, Error> {
        if !self.is_square() {
            panic!("Trying to invert a non-square matrix")
        }

        let row_count = self.row_count;
        let col_count = self.col_count;

        let mut work = self.augment(&Self::identity(row_count));
        work.gaussian_elim()?;

        Ok(work.sub_matrix(0, row_count, col_count, col_count * 2))
    }

    pub fn vandermonde(rows: usize, cols: usize) -> Matrix<F> {
        let mut result = Self::new(rows, cols);

        for r in 0..rows {
            // doesn't matter what `r_a` is as long as it's unique.
            // then the vandermonde matrix is invertible.
            let r_a = F::nth(r);
            for c in 0..cols {
                acc!(result, r, c) = F::exp(r_a, c);
            }
        }

        result
    }
}

impl<F: Field> fmt::Display for Matrix<F> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("[\n")?;
        for i in 0..self.row_count {
            fmt.write_fmt(format_args!(
                " {:?}\n",
                &self.data[(i * self.col_count)..(i * self.col_count + self.col_count)]
            ))?;
        }
        fmt.write_str("]\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galois::GF8Field;

    #[test]
    fn test_print_matrix() {
        let mat: Matrix<GF8Field> = Matrix::identity(3);
        println!("{}", mat);
    }
}
