use atlas_ndarray::Numeric;
use num_traits::Float;

pub(crate) fn dot_slice<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let len = lhs.len();
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();
    let mut index = 0;

    while index + 4 <= len {
        acc0 += lhs[index] * rhs[index];
        acc1 += lhs[index + 1] * rhs[index + 1];
        acc2 += lhs[index + 2] * rhs[index + 2];
        acc3 += lhs[index + 3] * rhs[index + 3];
        index += 4;
    }

    let mut total = acc0 + acc1;
    total += acc2 + acc3;

    while index < len {
        total += lhs[index] * rhs[index];
        index += 1;
    }

    total
}

pub(crate) fn vector_norm<T: Numeric + Float>(values: &[T]) -> T {
    dot_slice(values, values).sqrt()
}

pub(crate) fn is_symmetric<T: Float>(matrix: &[T], n: usize, tolerance: T) -> bool {
    for row in 0..n {
        for col in 0..row {
            let diff = (matrix[row * n + col] - matrix[col * n + row]).abs();

            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

pub(crate) fn tolerance<T: Float>() -> T {
    T::epsilon().sqrt()
}
