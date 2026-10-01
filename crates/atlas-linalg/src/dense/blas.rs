use atlas_ndarray::Numeric;

use crate::internal::{
    dense::{MatrixRef, VectorRef},
    simd,
};

pub(crate) fn dot<T: Numeric>(lhs: &[T], rhs: &[T]) -> Option<T> {
    if simd::is_f32::<T>() {
        return atlas_blas::dot_f32(
            simd::cast_slice::<T, f32>(lhs),
            simd::cast_slice::<T, f32>(rhs),
        )
        .map(simd::cast_value);
    }
    if simd::is_f64::<T>() {
        return atlas_blas::dot_f64(
            simd::cast_slice::<T, f64>(lhs),
            simd::cast_slice::<T, f64>(rhs),
        )
        .map(simd::cast_value);
    }

    None
}

pub(crate) fn matrix_vector<T: Numeric>(
    matrix: MatrixRef<'_, T>,
    vector: VectorRef<'_, T>,
) -> Option<Vec<T>> {
    let mut output = vec![T::zero(); matrix.rows];

    if simd::is_f32::<T>() {
        atlas_blas::matrix_vector_f32(
            matrix_f32(matrix)?,
            simd::cast_slice::<T, f32>(vector.contiguous_slice()),
            simd::cast_mut_slice::<T, f32>(&mut output),
        )?;
        return Some(output);
    }
    if simd::is_f64::<T>() {
        atlas_blas::matrix_vector_f64(
            matrix_f64(matrix)?,
            simd::cast_slice::<T, f64>(vector.contiguous_slice()),
            simd::cast_mut_slice::<T, f64>(&mut output),
        )?;
        return Some(output);
    }

    None
}

pub(crate) fn vector_matrix<T: Numeric>(
    vector: VectorRef<'_, T>,
    matrix: MatrixRef<'_, T>,
) -> Option<Vec<T>> {
    let mut output = vec![T::zero(); matrix.cols];

    if simd::is_f32::<T>() {
        atlas_blas::vector_matrix_f32(
            simd::cast_slice::<T, f32>(vector.contiguous_slice()),
            matrix_f32(matrix)?,
            simd::cast_mut_slice::<T, f32>(&mut output),
        )?;
        return Some(output);
    }
    if simd::is_f64::<T>() {
        atlas_blas::vector_matrix_f64(
            simd::cast_slice::<T, f64>(vector.contiguous_slice()),
            matrix_f64(matrix)?,
            simd::cast_mut_slice::<T, f64>(&mut output),
        )?;
        return Some(output);
    }

    None
}

pub(crate) fn matrix_matrix<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Option<Vec<T>> {
    let mut output = vec![T::zero(); lhs.rows.checked_mul(rhs.cols)?];

    if simd::is_f32::<T>() {
        atlas_blas::matrix_matrix_f32(
            matrix_f32(lhs)?,
            matrix_f32(rhs)?,
            simd::cast_mut_slice::<T, f32>(&mut output),
        )?;
        return Some(output);
    }
    if simd::is_f64::<T>() {
        atlas_blas::matrix_matrix_f64(
            matrix_f64(lhs)?,
            matrix_f64(rhs)?,
            simd::cast_mut_slice::<T, f64>(&mut output),
        )?;
        return Some(output);
    }

    None
}

fn matrix_f32<'a, T: Numeric>(matrix: MatrixRef<'a, T>) -> Option<atlas_blas::Matrix<'a, f32>> {
    atlas_blas::Matrix::new(
        simd::cast_slice::<T, f32>(matrix.data),
        matrix.offset,
        matrix.rows,
        matrix.cols,
        matrix.row_stride,
        matrix.col_stride,
    )
}

fn matrix_f64<'a, T: Numeric>(matrix: MatrixRef<'a, T>) -> Option<atlas_blas::Matrix<'a, f64>> {
    atlas_blas::Matrix::new(
        simd::cast_slice::<T, f64>(matrix.data),
        matrix.offset,
        matrix.rows,
        matrix.cols,
        matrix.row_stride,
        matrix.col_stride,
    )
}
