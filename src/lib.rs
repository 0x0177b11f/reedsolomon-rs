#[cfg(test)]
extern crate rand;

#[cfg(test)]
mod tests;

mod errors;

#[macro_use]
mod macros;

mod field;
mod galois;
mod matrix;
mod inversion_tree;
mod reedsolomon;
pub use crate::reedsolomon::ReedSolomon;
