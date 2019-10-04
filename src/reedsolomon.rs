use crate::field::Field;
use crate::galois::GF8Field;
use crate::matrix::Matrix;
use crate::inversion_tree::InversionTree;



#[derive(Debug)]
struct ReedSolomonImpl<F: Field> {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,
    matrix: Matrix<F>,
    tree: InversionTree<F>,
}



pub type ReedSolomon = ReedSolomonImpl<GF8Field>;