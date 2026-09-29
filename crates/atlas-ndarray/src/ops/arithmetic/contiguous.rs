use rayon::prelude::*;

use super::{ElementwiseArithmetic, ElementwiseDivision, from_owned_parts};
use crate::{
    NDArray, Numeric,
    internal::{
        parallel::{ELEMENTWISE_CHUNK_LEN, should_parallelize_elementwise},
        simd,
    },
};

fn apply_contiguous<T, F>(lhs: &[T], rhs: &[T], out: &mut [T], kernel: F)
where
    T: Numeric,
    F: Fn(&[T], &[T], &mut [T]) + Sync,
{
    if should_parallelize_elementwise(out.len()) {
        out.par_chunks_mut(ELEMENTWISE_CHUNK_LEN)
            .zip(lhs.par_chunks(ELEMENTWISE_CHUNK_LEN))
            .zip(rhs.par_chunks(ELEMENTWISE_CHUNK_LEN))
            .for_each(|((out, lhs), rhs)| kernel(lhs, rhs, out));
    } else {
        kernel(lhs, rhs, out);
    }
}

pub(super) fn elementwise_binary_contiguous<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    simd::map_binary_contiguous(lhs.data(), rhs.data(), &mut data, op);

    from_owned_parts(lhs.shape().to_vec(), data)
}

pub(super) fn elementwise_add_contiguous<T: ElementwiseArithmetic>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> NDArray<T> {
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    apply_contiguous(lhs.data(), rhs.data(), &mut data, simd::add_contiguous);

    from_owned_parts(lhs.shape().to_vec(), data)
}

pub(super) fn elementwise_mul_contiguous<T: ElementwiseArithmetic>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> NDArray<T> {
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    apply_contiguous(lhs.data(), rhs.data(), &mut data, simd::mul_contiguous);

    from_owned_parts(lhs.shape().to_vec(), data)
}

pub(super) fn elementwise_sub_contiguous<T: ElementwiseArithmetic>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> NDArray<T> {
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    apply_contiguous(lhs.data(), rhs.data(), &mut data, simd::sub_contiguous);

    from_owned_parts(lhs.shape().to_vec(), data)
}

pub(super) fn elementwise_div_contiguous<T: ElementwiseDivision>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> NDArray<T> {
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    apply_contiguous(lhs.data(), rhs.data(), &mut data, simd::div_contiguous);

    from_owned_parts(lhs.shape().to_vec(), data)
}
