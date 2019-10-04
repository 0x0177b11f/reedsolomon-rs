use crate::galois::*;


/// A finite field to perform encoding over.
pub trait Field: Sized {
    /// The order of the field. This is a limit on the number of shards
    /// in an encoding.
    const ORDER: usize;

    /// The representational type of the field.
    type Elem: Default + Clone + Copy + PartialEq + std::fmt::Debug;

    /// Add two elements together.
    fn add(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Multiply two elements together.
    fn mul(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Divide a by b. Panics is b is zero.
    fn div(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// Raise `a` to the n'th power.
    fn exp(a: Self::Elem, n: usize) -> Self::Elem;

    /// The "zero" element or additive identity.
    fn zero() -> Self::Elem;

    /// The "one" element or multiplicative identity.
    fn one() -> Self::Elem;

    fn nth_internal(n: usize) -> Self::Elem;

    /// Yield the nth element of the field. Panics if n >= ORDER.
    /// Assignment is arbitrary but must be unique to `n`.
    fn nth(n: usize) -> Self::Elem {
        if n >= Self::ORDER {
            let pow = (Self::ORDER as f32).log(2.0) as usize;
            panic!("{} out of bounds for GF(2^{}) member", n, pow)
        }

        Self::nth_internal(n)
    }

    /// Multiply a slice of elements by another. Writes into the output slice.
    ///
    /// # Panics
    /// Panics if the output slice does not have equal length to the input.
    fn mul_slice(elem: Self::Elem, input: &[Self::Elem], out: &mut [Self::Elem]) {
        assert_eq!(input.len(), out.len());

        for (i, o) in input.iter().zip(out) {
            *o = Self::mul(elem.clone(), i.clone())
        }
    }

    /// Multiply a slice of elements by another, adding each result to the corresponding value in
    /// `out`.
    ///
    /// # Panics
    /// Panics if the output slice does not have equal length to the input.
    fn mul_slice_add(elem: Self::Elem, input: &[Self::Elem], out: &mut [Self::Elem]) {
        assert_eq!(input.len(), out.len());

        for (i, o) in input.iter().zip(out) {
            *o = Self::add(o.clone(), Self::mul(elem.clone(), i.clone()))
        }
    }
}


