use crate::{NDArray, Numeric, internal::simd};

use super::from_owned_parts;

pub(super) fn add_scalar_rhs<T: Numeric>(array: &NDArray<T>, scalar: T) -> NDArray<T> {
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    simd::add_scalar_contiguous(array.data(), scalar, &mut data);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn add_scalar_lhs<T: Numeric>(scalar: T, array: &NDArray<T>) -> NDArray<T> {
    add_scalar_rhs(array, scalar)
}

pub(super) fn mul_scalar_rhs<T: Numeric>(array: &NDArray<T>, scalar: T) -> NDArray<T> {
    let len = array.data().len();
    let mut data = vec![T::zero(); len];
    simd::mul_scalar_contiguous(array.data(), scalar, &mut data);

    from_owned_parts(array.shape().to_vec(), data)
}

pub(super) fn mul_scalar_lhs<T: Numeric>(scalar: T, array: &NDArray<T>) -> NDArray<T> {
    mul_scalar_rhs(array, scalar)
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
