use rayon::prelude::*;

use super::{ElementwiseArithmetic, ElementwiseDivision, from_owned_parts};
use crate::{
    NDArray, Numeric,
    internal::{
        parallel::{ELEMENTWISE_CHUNK_LEN, should_parallelize_elementwise},
        simd,
    },
};

fn apply_scalar<T, F>(input: &[T], scalar: T, out: &mut [T], kernel: F)
where
    T: Numeric,
    F: Fn(&[T], T, &mut [T]) + Sync,
{
    if should_parallelize_elementwise(out.len()) {
        out.par_chunks_mut(ELEMENTWISE_CHUNK_LEN)
            .zip(input.par_chunks(ELEMENTWISE_CHUNK_LEN))
            .for_each(|(out, input)| kernel(input, scalar, out));
    } else {
        kernel(input, scalar, out);
    }
}

pub(super) fn add_scalar_rhs<T: ElementwiseArithmetic>(
    array: &NDArray<T>,
    scalar: T,
) -> NDArray<T> {
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    apply_scalar(array.data(), scalar, &mut data, simd::add_scalar_contiguous);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn add_scalar_lhs<T: ElementwiseArithmetic>(
    scalar: T,
    array: &NDArray<T>,
) -> NDArray<T> {
    add_scalar_rhs(array, scalar)
}

pub(super) fn mul_scalar_rhs<T: ElementwiseArithmetic>(
    array: &NDArray<T>,
    scalar: T,
) -> NDArray<T> {
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    apply_scalar(array.data(), scalar, &mut data, simd::mul_scalar_contiguous);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn mul_scalar_lhs<T: ElementwiseArithmetic>(
    scalar: T,
    array: &NDArray<T>,
) -> NDArray<T> {
    mul_scalar_rhs(array, scalar)
}

pub(super) fn sub_scalar_rhs<T: ElementwiseArithmetic>(
    array: &NDArray<T>,
    scalar: T,
) -> NDArray<T> {
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    apply_scalar(array.data(), scalar, &mut data, simd::sub_scalar_contiguous);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn div_scalar_rhs<T: ElementwiseDivision>(array: &NDArray<T>, scalar: T) -> NDArray<T> {
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    apply_scalar(array.data(), scalar, &mut data, simd::div_scalar_contiguous);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn elementwise_scalar_rhs<T, F>(array: &NDArray<T>, scalar: T, op: F) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    simd::map_scalar_contiguous(array.data(), scalar, &mut data, op);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn elementwise_scalar_lhs<T, F>(scalar: T, array: &NDArray<T>, op: F) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    simd::map_scalar_contiguous(array.data(), scalar, &mut data, |value, rhs| op(rhs, value));

    from_owned_parts(array.shape().to_vec(), data)
}
