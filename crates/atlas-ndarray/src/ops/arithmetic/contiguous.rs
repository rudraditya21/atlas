use crate::{NDArray, Numeric, internal::simd};

use super::from_owned_parts;

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

pub(super) fn elementwise_add_contiguous<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> NDArray<T> {
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    simd::add_contiguous(lhs.data(), rhs.data(), &mut data);

    from_owned_parts(lhs.shape().to_vec(), data)
}

pub(super) fn elementwise_mul_contiguous<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> NDArray<T> {
    let len = lhs.data().len();
    let mut data = vec![T::zero(); len];
    simd::mul_contiguous(lhs.data(), rhs.data(), &mut data);

    from_owned_parts(lhs.shape().to_vec(), data)
}
