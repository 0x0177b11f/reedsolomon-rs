use crate::errors::Error;
use crate::field::{Field, ReconstructShard};
use crate::galois::GF8Field;
use crate::inversion_tree::InversionTree;
use crate::matrix::Matrix;

use smallvec::SmallVec;
use std::sync::Arc;
use std::fmt;

#[derive(Debug)]
pub struct ReedSolomonImpl<F: Field> {
    m_data_shard_count: usize,
    m_parity_shard_count: usize,
    m_total_shard_count: usize,
    matrix: Matrix<F>,
    tree: InversionTree<F>,
}

impl<F: Field> Clone for ReedSolomonImpl<F> {
    fn clone(&self) -> ReedSolomonImpl<F> {
        ReedSolomonImpl::new(self.data_shard_count(), self.parity_shard_count())
            .expect("basic checks already passed as precondition of existence of self")
    }
}

impl<F: Field> PartialEq for ReedSolomonImpl<F> {
    fn eq(&self, rhs: &ReedSolomonImpl<F>) -> bool {
        self.data_shard_count() == rhs.data_shard_count()
            && self.parity_shard_count() == rhs.parity_shard_count()
    }
}

impl<F: Field> fmt::Display for ReedSolomonImpl<F> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_fmt(format_args!("data shard: {} \n", self.data_shard_count()))?;
        fmt.write_fmt(format_args!("parity shard: {} \n", self.parity_shard_count()))?;
        fmt.write_fmt(format_args!("total shard: {} \n", self.total_shard_count()))?;
        self.matrix.fmt(fmt)?;
        Ok(())
    }
}


impl<F: Field> ReedSolomonImpl<F> {
    // AUDIT
    //
    // Error detection responsibilities
    //
    // Terminologies and symbols:
    //   X =A, B, C=> Y: X delegates error checking responsibilities A, B, C to Y
    //   X:= A, B, C: X needs to handle responsibilities A, B, C
    //
    // Encode methods
    //
    // `encode_single`:=
    //   - check index `i_data` within range [0, data shard count)
    //   - check length of `slices` matches total shard count exactly
    //   - check consistency of length of individual slices
    // `encode_single_sep`:=
    //   - check index `i_data` within range [0, data shard count)
    //   - check length of `parity` matches parity shard count exactly
    //   - check consistency of length of individual parity slices
    //   - check length of `single_data` matches length of first parity slice
    // `encode`:=
    //   - check length of `slices` matches total shard count exactly
    //   - check consistency of length of individual slices
    // `encode_sep`:=
    //   - check length of `data` matches data shard count exactly
    //   - check length of `parity` matches parity shard count exactly
    //   - check consistency of length of individual data slices
    //   - check consistency of length of individual parity slices
    //   - check length of first parity slice matches length of first data slice
    //
    // Verify methods
    //
    // `verify`:=
    //   - check length of `slices` matches total shard count exactly
    //   - check consistency of length of individual slices
    //
    //   Generates buffer then passes control to verify_with_buffer
    //
    // `verify_with_buffer`:=
    //   - check length of `slices` matches total shard count exactly
    //   - check length of `buffer` matches parity shard count exactly
    //   - check consistency of length of individual slices
    //   - check consistency of length of individual slices in buffer
    //   - check length of first slice in buffer matches length of first slice
    //
    // Reconstruct methods
    //
    // `reconstruct` =ALL=> `reconstruct_internal`
    // `reconstruct_data`=ALL=> `reconstruct_internal`
    // `reconstruct_internal`:=
    //   - check length of `slices` matches total shard count exactly
    //   - check consistency of length of individual slices
    //   - check length of `slice_present` matches length of `slices`

    fn get_parity_rows(&self) -> SmallVec<[&[F::Elem]; 32]> {
        let mut parity_rows = SmallVec::with_capacity(self.parity_shard_count());
        let matrix = &self.matrix;
        for i in self.data_shard_count()..self.total_shard_count() {
            parity_rows.push(matrix.get_row(i));
        }

        parity_rows
    }

    fn build_matrix(data_shards: usize, total_shards: usize) -> Matrix<F> {
        let vandermonde = Matrix::vandermonde(total_shards, data_shards);

        let top = vandermonde.sub_matrix(0, 0, data_shards, data_shards);

        vandermonde.multiply(&top.invert().unwrap())
    }

    /// Creates a new instance of Reed-Solomon erasure code encoder/decoder.
    ///
    /// Returns `Error::TooFewDataShards` if `data_shards == 0`.
    ///
    /// Returns `Error::TooFewParityShards` if `parity_shards == 0`.
    ///
    /// Returns `Error::TooManyShards` if `data_shards + parity_shards > F::ORDER`.
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<ReedSolomonImpl<F>, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > F::ORDER {
            return Err(Error::TooManyShards);
        }

        let total_shards = data_shards + parity_shards;

        let matrix = Self::build_matrix(data_shards, total_shards);

        Ok(ReedSolomonImpl {
            m_data_shard_count: data_shards,
            m_parity_shard_count: parity_shards,
            m_total_shard_count: total_shards,
            matrix,
            tree: InversionTree::new(data_shards, parity_shards),
        })
    }

    pub fn data_shard_count(&self) -> usize {
        self.m_data_shard_count
    }

    pub fn parity_shard_count(&self) -> usize {
        self.m_parity_shard_count
    }

    pub fn total_shard_count(&self) -> usize {
        self.m_total_shard_count
    }

    fn code_some_slices<T: AsRef<[F::Elem]>, U: AsMut<[F::Elem]>>(
        &self,
        matrix_rows: &[&[F::Elem]],
        inputs: &[T],
        outputs: &mut [U],
    ) {
        for i_input in 0..self.data_shard_count() {
            self.code_single_slice(matrix_rows, i_input, inputs[i_input].as_ref(), outputs);
        }
    }

    fn code_single_slice<U: AsMut<[F::Elem]>>(
        &self,
        matrix_rows: &[&[F::Elem]],
        i_input: usize,
        input: &[F::Elem],
        outputs: &mut [U],
    ) {
        outputs.iter_mut().enumerate().for_each(|(i_row, output)| {
            let matrix_row_to_use = matrix_rows[i_row][i_input];
            let output = output.as_mut();

            if i_input == 0 {
                F::mul_slice(matrix_row_to_use, input, output);
            } else {
                F::mul_slice_add(matrix_row_to_use, input, output);
            }
        })
    }

    fn check_some_slices_with_buffer<T, U>(
        &self,
        matrix_rows: &[&[F::Elem]],
        inputs: &[T],
        to_check: &[T],
        buffer: &mut [U],
    ) -> bool
    where
        T: AsRef<[F::Elem]>,
        U: AsRef<[F::Elem]> + AsMut<[F::Elem]>,
    {
        self.code_some_slices(matrix_rows, inputs, buffer);

        let at_least_one_mismatch_present = buffer
            .iter_mut()
            .enumerate()
            .map(|(i, expected_parity_shard)| {
                expected_parity_shard.as_ref() == to_check[i].as_ref()
            })
            .any(|x| !x); // find the first false (some slice is different from the expected one)
        !at_least_one_mismatch_present
    }

    /// Constructs the parity shards partially using only the data shard
    /// indexed by `i_data`.
    ///
    /// The slots where the parity shards sit at will be overwritten.
    ///
    /// # Warning
    ///
    /// You must apply this method on the data shards in strict sequential order (0..data shard count),
    /// otherwise the parity shards will be incorrect.
    ///
    /// It is recommended to use the `ShardByShard` bookkeeping struct instead of this method directly.
    pub fn encode_single<T, U>(&self, i_data: usize, mut shards: T) -> Result<(), Error>
    where
        T: AsRef<[U]> + AsMut<[U]>,
        U: AsRef<[F::Elem]> + AsMut<[F::Elem]>,
    {
        let slices = shards.as_mut();

        check_slice_index!(data => self, i_data);
        check_piece_count!(all=> self, slices);
        check_slices!(multi => slices);

        // Get the slice of output buffers.
        let (mut_input, output) = slices.split_at_mut(self.data_shard_count());

        let input = mut_input[i_data].as_ref();

        self.encode_single_sep(i_data, input, output)
    }

    /// Constructs the parity shards partially using only the data shard provided.
    ///
    /// The data shard must match the index `i_data`.
    ///
    /// The slots where the parity shards sit at will be overwritten.
    ///
    /// # Warning
    ///
    /// You must apply this method on the data shards in strict sequential order (0..data shard count),
    /// otherwise the parity shards will be incorrect.
    ///
    /// It is recommended to use the `ShardByShard` bookkeeping struct instead of this method directly.
    pub fn encode_single_sep<U: AsRef<[F::Elem]> + AsMut<[F::Elem]>>(
        &self,
        i_data: usize,
        single_data: &[F::Elem],
        parity: &mut [U],
    ) -> Result<(), Error> {
        check_slice_index!(data => self, i_data);
        check_piece_count!(parity => self, parity);
        check_slices!(multi => parity, single => single_data);

        let parity_rows = self.get_parity_rows();

        // Do the coding.
        self.code_single_slice(&parity_rows, i_data, single_data, parity);

        Ok(())
    }

    /// Constructs the parity shards.
    ///
    /// The slots where the parity shards sit at will be overwritten.
    pub fn encode<T, U>(&self, mut shards: T) -> Result<(), Error>
    where
        T: AsRef<[U]> + AsMut<[U]>,
        U: AsRef<[F::Elem]> + AsMut<[F::Elem]>,
    {
        let slices: &mut [U] = shards.as_mut();

        check_piece_count!(all => self, slices);
        check_slices!(multi => slices);

        // Get the slice of output buffers.
        let (input, output) = slices.split_at_mut(self.data_shard_count());

        self.encode_sep(&*input, output)
    }

    /// Constructs the parity shards using a read-only view into the
    /// data shards.
    ///
    /// The slots where the parity shards sit at will be overwritten.
    pub fn encode_sep<T: AsRef<[F::Elem]>, U: AsRef<[F::Elem]> + AsMut<[F::Elem]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        check_piece_count!(data => self, data);
        check_piece_count!(parity => self, parity);
        check_slices!(multi => data, multi => parity);

        let parity_rows = self.get_parity_rows();

        // Do the coding.
        self.code_some_slices(&parity_rows, data, parity);

        Ok(())
    }

    /// Checks if the parity shards are correct.
    ///
    /// This is a wrapper of `verify_with_buffer`.
    pub fn verify<T: AsRef<[F::Elem]>>(&self, slices: &[T]) -> Result<bool, Error> {
        check_piece_count!(all => self, slices);
        check_slices!(multi => slices);

        let slice_len = slices[0].as_ref().len();

        let mut buffer: SmallVec<[Vec<F::Elem>; 32]> =
            SmallVec::with_capacity(self.parity_shard_count());

        for _ in 0..self.parity_shard_count() {
            buffer.push(vec![F::zero(); slice_len]);
        }

        self.verify_with_buffer(slices, &mut buffer)
    }

    /// Checks if the parity shards are correct.
    pub fn verify_with_buffer<T, U>(&self, slices: &[T], buffer: &mut [U]) -> Result<bool, Error>
    where
        T: AsRef<[F::Elem]>,
        U: AsRef<[F::Elem]> + AsMut<[F::Elem]>,
    {
        check_piece_count!(all => self, slices);
        check_piece_count!(parity_buf => self, buffer);
        check_slices!(multi => slices, multi => buffer);

        let data = &slices[0..self.data_shard_count()];
        let to_check = &slices[self.data_shard_count()..];

        let parity_rows = self.get_parity_rows();

        Ok(self.check_some_slices_with_buffer(&parity_rows, data, to_check, buffer))
    }

    /// Reconstructs all shards.
    ///
    /// The shards marked not present are only overwritten when no error
    /// is detected. All provided shards must have the same length.
    ///
    /// This means if the method returns an `Error`, then nothing is touched.
    ///
    /// `reconstruct`, `reconstruct_data`, `reconstruct_shards`,
    /// `reconstruct_data_shards` share the same core code base.
    pub fn reconstruct<T: ReconstructShard<F>>(&self, slices: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(slices, false)
    }

    /// Reconstructs only the data shards.
    ///
    /// The shards marked not present are only overwritten when no error
    /// is detected. All provided shards must have the same length.
    ///
    /// This means if the method returns an `Error`, then nothing is touched.
    ///
    /// `reconstruct`, `reconstruct_data`, `reconstruct_shards`,
    /// `reconstruct_data_shards` share the same core code base.
    pub fn reconstruct_data<T: ReconstructShard<F>>(&self, slices: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(slices, true)
    }

    fn get_data_decode_matrix(
        &self,
        valid_indices: &[usize],
        invalid_indices: &[usize],
    ) -> Arc<Matrix<F>> {
        // Attempt to get the cached inverted matrix out of the tree
        // based on the indices of the invalid rows.
        match self.tree.get_inverted_matrix(&invalid_indices) {
            // If the inverted matrix isn't cached in the tree yet we must
            // construct it ourselves and insert it into the tree for the
            // future.  In this way the inversion tree is lazily loaded.
            None => {
                // Pull out the rows of the matrix that correspond to the
                // shards that we have and build a square matrix.  This
                // matrix could be used to generate the shards that we have
                // from the original data.
                let mut sub_matrix = Matrix::new(self.data_shard_count(), self.data_shard_count());
                for (sub_matrix_row, &valid_index) in valid_indices.into_iter().enumerate() {
                    for c in 0..self.data_shard_count() {
                        sub_matrix.set(sub_matrix_row, c, self.matrix.get(valid_index, c));
                    }
                }
                // Invert the matrix, so we can go from the encoded shards
                // back to the original data.  Then pull out the row that
                // generates the shard that we want to decode.  Note that
                // since this matrix maps back to the original data, it can
                // be used to create a data shard, but not a parity shard.
                let data_decode_matrix = Arc::new(sub_matrix.invert().unwrap());

                // Cache the inverted matrix in the tree for future use keyed on the
                // indices of the invalid rows.
                self.tree
                    .insert_inverted_matrix(&invalid_indices, &data_decode_matrix)
                    .unwrap();

                data_decode_matrix
            }
            Some(m) => m,
        }
    }

    fn reconstruct_internal<T: ReconstructShard<F>>(
        &self,
        shards: &mut [T],
        data_only: bool,
    ) -> Result<(), Error> {
        check_piece_count!(all => self, shards);

        let data_shard_count = self.data_shard_count();

        // Quick check: are all of the shards present?  If so, there's
        // nothing to do.
        let mut number_present = 0;
        let mut shard_len = None;

        for shard in shards.iter_mut() {
            if let Some(len) = shard.len() {
                if len == 0 {
                    return Err(Error::EmptyShard);
                }
                number_present += 1;
                if let Some(old_len) = shard_len {
                    if len != old_len {
                        // mismatch between shards.
                        return Err(Error::IncorrectShardSize);
                    }
                }
                shard_len = Some(len);
            }
        }

        if number_present == self.total_shard_count() {
            // Cool.  All of the shards are there.  We don't
            // need to do anything.
            return Ok(());
        }

        // More complete sanity check
        if number_present < data_shard_count {
            return Err(Error::TooFewShardsPresent);
        }

        let shard_len = shard_len.expect("at least one shard present; qed");

        // Pull out an array holding just the shards that
        // correspond to the rows of the submatrix.  These shards
        // will be the input to the decoding process that re-creates
        // the missing data shards.
        //
        // Also, create an array of indices of the valid rows we do have
        // and the invalid rows we don't have.
        //
        // The valid indices are used to construct the data decode matrix,
        // the invalid indices are used to key the data decode matrix
        // in the inversion tree.
        //
        // We only need exactly N valid indices, where N = `data_shard_count()`,
        // as the data decode matrix is a N x N matrix, thus only needs
        // N valid indices for determining the N rows to pick from
        // `self.matrix`.
        let mut sub_shards: SmallVec<[&[F::Elem]; 32]> = SmallVec::with_capacity(data_shard_count);
        let mut missing_data_slices: SmallVec<[&mut [F::Elem]; 32]> =
            SmallVec::with_capacity(self.parity_shard_count());
        let mut missing_parity_slices: SmallVec<[&mut [F::Elem]; 32]> =
            SmallVec::with_capacity(self.parity_shard_count());
        let mut valid_indices: SmallVec<[usize; 32]> = SmallVec::with_capacity(data_shard_count);
        let mut invalid_indices: SmallVec<[usize; 32]> = SmallVec::with_capacity(data_shard_count);

        // Separate the shards into groups
        for (matrix_row, shard) in shards.into_iter().enumerate() {
            // get or initialize the shard so we can reconstruct in-place,
            // but if we are only reconstructing data shard,
            // do not initialize if the shard is not a data shard
            let shard_data = if matrix_row >= data_shard_count && data_only {
                shard.get().ok_or(None)
            } else {
                shard.get_or_initialize(shard_len).map_err(Some)
            };

            match shard_data {
                Ok(shard) => {
                    if sub_shards.len() < data_shard_count {
                        sub_shards.push(shard);
                        valid_indices.push(matrix_row);
                    } else {
                        // Already have enough shards in `sub_shards`
                        // as we only need N shards, where N = `data_shard_count()`,
                        // for the data decode matrix
                        //
                        // So nothing to do here
                    }
                }
                Err(None) => {
                    // the shard data is not meant to be initialized here,
                    // but we should still note it missing.
                    invalid_indices.push(matrix_row);
                }
                Err(Some(x)) => {
                    // initialized missing shard data.
                    let shard = x?;
                    if matrix_row < data_shard_count {
                        missing_data_slices.push(shard);
                    } else {
                        missing_parity_slices.push(shard);
                    }

                    invalid_indices.push(matrix_row);
                }
            }
        }

        let data_decode_matrix = self.get_data_decode_matrix(&valid_indices, &invalid_indices);

        // Re-create any data shards that were missing.
        //
        // The input to the coding is all of the shards we actually
        // have, and the output is the missing data shards. The computation
        // is done using the special decode matrix we just built.
        let mut matrix_rows: SmallVec<[&[F::Elem]; 32]> =
            SmallVec::with_capacity(self.parity_shard_count());

        for i_slice in invalid_indices
            .iter()
            .cloned()
            .take_while(|i| i < &data_shard_count)
        {
            matrix_rows.push(data_decode_matrix.get_row(i_slice));
        }

        self.code_some_slices(&matrix_rows, &sub_shards, &mut missing_data_slices);

        if data_only {
            Ok(())
        } else {
            // Now that we have all of the data shards intact, we can
            // compute any of the parity that is missing.
            //
            // The input to the coding is ALL of the data shards, including
            // any that we just calculated.  The output is whichever of the
            // parity shards were missing.
            let mut matrix_rows: SmallVec<[&[F::Elem]; 32]> =
                SmallVec::with_capacity(self.parity_shard_count());
            let parity_rows = self.get_parity_rows();

            for i_slice in invalid_indices
                .iter()
                .cloned()
                .skip_while(|i| i < &data_shard_count)
            {
                matrix_rows.push(parity_rows[i_slice - data_shard_count]);
            }
            {
                // Gather up all the data shards.
                // old data shards are in `sub_shards`,
                // new ones are in `missing_data_slices`.
                let mut i_old_data_slice = 0;
                let mut i_new_data_slice = 0;

                let mut all_data_slices: SmallVec<[&[F::Elem]; 32]> =
                    SmallVec::with_capacity(data_shard_count);

                let mut next_maybe_good = 0;
                let mut push_good_up_to = move |data_slices: &mut SmallVec<_>, up_to| {
                    // if next_maybe_good == up_to, this loop is a no-op.
                    for _ in next_maybe_good..up_to {
                        // push all good indices we just skipped.
                        data_slices.push(sub_shards[i_old_data_slice]);
                        i_old_data_slice += 1;
                    }

                    next_maybe_good = up_to + 1;
                };

                for i_slice in invalid_indices
                    .iter()
                    .cloned()
                    .take_while(|i| i < &data_shard_count)
                {
                    push_good_up_to(&mut all_data_slices, i_slice);
                    all_data_slices.push(missing_data_slices[i_new_data_slice]);
                    i_new_data_slice += 1;
                }
                push_good_up_to(&mut all_data_slices, data_shard_count);

                // Now do the actual computation for the missing
                // parity shards
                self.code_some_slices(&matrix_rows, &all_data_slices, &mut missing_parity_slices);
            }

            Ok(())
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct ShardByShard<'a, F: 'a + Field> {
    codec: &'a ReedSolomonImpl<F>,
    cur_input: usize,
}

pub type ReedSolomon = ReedSolomonImpl<GF8Field>;
