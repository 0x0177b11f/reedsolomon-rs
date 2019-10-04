#[cfg(test)]
extern crate rand;

mod errors;

#[macro_use]
mod macros;

mod field;
mod galois;
mod inversion_tree;
mod matrix;
mod reedsolomon;
pub use crate::reedsolomon::ReedSolomon;
