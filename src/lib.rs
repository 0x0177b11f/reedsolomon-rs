#[cfg(test)]
extern crate rand;

#[cfg(test)]
mod tests;

#[macro_use]
extern crate error_chain;

mod error;

mod field;
mod galois;

mod matrix;
mod inversion_tree;
mod reedsolomon;
pub use crate::reedsolomon::ReedSolomon;
