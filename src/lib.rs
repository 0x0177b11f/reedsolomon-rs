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

#[cfg(test)]
mod tests {
    use super::ReedSolomon;
    use rand::{thread_rng, Rng};
    
    #[test]
    fn test_reedsolomon() {
        let data_row = 8;  // 8 data shards,
        let parity_row = 2; // 2 parity shards

        let r = ReedSolomon::new(data_row, parity_row).unwrap();

        let mut master_copy: Vec<Vec<u8>> = vec![vec![0u8; 512]; data_row + parity_row];
        for i in 0..data_row {
            thread_rng().try_fill(&mut master_copy[i][..]).unwrap();
        }

        // Construct the parity shards
        r.encode(&mut master_copy).unwrap();

        // Make a copy and transform it into option shards arrangement
        // for feeding into reconstruct_shards
        let mut shards: Vec<_> = master_copy.clone().into_iter().map(Some).collect();

        // We can remove up to 2 shards, which may be data or parity shards
        shards[0] = None;
        shards[1] = None;

        // Try to reconstruct missing shards
        r.reconstruct(&mut shards).unwrap();

        // Convert back to normal shard arrangement
        let result: Vec<_> = shards.into_iter().filter_map(|x| x).collect();

        assert!(r.verify(&result).unwrap());
        assert_eq!(master_copy, result);
    }
}
