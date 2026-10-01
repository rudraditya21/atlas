use atlas_blas::{Layout, Transpose};
use atlas_ndarray::Numeric;

use crate::internal::{
    dense::{MatrixRef, VectorRef},
    simd,
};

pub(crate) fn dot<T: Numeric>(lhs: &[T], rhs: &[T]) -> Option<T> {
    if simd::is_f32::<T>() {
        return atlas_blas::dot_f32(
            simd::cast_slice::<T, f32>(lhs),
            1,
            simd::cast_slice::<T, f32>(rhs),
            1,
            lhs.len(),
        )
        .ok()
        .map(simd::cast_value);
    }
    if simd::is_f64::<T>() {
        return atlas_blas::dot_f64(
            simd::cast_slice::<T, f64>(lhs),
            1,
            simd::cast_slice::<T, f64>(rhs),
            1,
            lhs.len(),
        )
        .ok()
        .map(simd::cast_value);
    }

    None
}

pub(crate) fn matrix_vector<T: Numeric>(
    matrix: MatrixRef<'_, T>,
    vector: VectorRef<'_, T>,
) -> Option<Vec<T>> {
    let (transpose, stored_rows, stored_cols, leading_dimension) = matrix_parameters(matrix)?;
    let mut output = vec![T::zero(); matrix.rows];

    if simd::is_f32::<T>() {
        atlas_blas::gemv_f32(
            Layout::RowMajor,
            transpose,
            stored_rows,
            stored_cols,
            1.0,
            simd::cast_slice::<T, f32>(&matrix.data[matrix.offset..]),
            leading_dimension,
            simd::cast_slice::<T, f32>(vector.contiguous_slice()),
            1,
            0.0,
            simd::cast_mut_slice::<T, f32>(&mut output),
            1,
        )
        .ok()?;
        return Some(output);
    }
    if simd::is_f64::<T>() {
        atlas_blas::gemv_f64(
            Layout::RowMajor,
            transpose,
            stored_rows,
            stored_cols,
            1.0,
            simd::cast_slice::<T, f64>(&matrix.data[matrix.offset..]),
            leading_dimension,
            simd::cast_slice::<T, f64>(vector.contiguous_slice()),
            1,
            0.0,
            simd::cast_mut_slice::<T, f64>(&mut output),
            1,
        )
        .ok()?;
        return Some(output);
    }

    None
}

pub(crate) fn vector_matrix<T: Numeric>(
    vector: VectorRef<'_, T>,
    matrix: MatrixRef<'_, T>,
) -> Option<Vec<T>> {
    let (matrix_transpose, stored_rows, stored_cols, leading_dimension) =
        matrix_parameters(matrix)?;
    let transpose = match matrix_transpose {
        Transpose::None => Transpose::Transpose,
        Transpose::Transpose => Transpose::None,
    };
    let mut output = vec![T::zero(); matrix.cols];

    if simd::is_f32::<T>() {
        atlas_blas::gemv_f32(
            Layout::RowMajor,
            transpose,
            stored_rows,
            stored_cols,
            1.0,
            simd::cast_slice::<T, f32>(&matrix.data[matrix.offset..]),
            leading_dimension,
            simd::cast_slice::<T, f32>(vector.contiguous_slice()),
            1,
            0.0,
            simd::cast_mut_slice::<T, f32>(&mut output),
            1,
        )
        .ok()?;
        return Some(output);
    }
    if simd::is_f64::<T>() {
        atlas_blas::gemv_f64(
            Layout::RowMajor,
            transpose,
            stored_rows,
            stored_cols,
            1.0,
            simd::cast_slice::<T, f64>(&matrix.data[matrix.offset..]),
            leading_dimension,
            simd::cast_slice::<T, f64>(vector.contiguous_slice()),
            1,
            0.0,
            simd::cast_mut_slice::<T, f64>(&mut output),
            1,
        )
        .ok()?;
        return Some(output);
    }

    None
}

pub(crate) fn matrix_matrix<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Option<Vec<T>> {
    let (lhs_transpose, _, _, lhs_leading_dimension) = matrix_parameters(lhs)?;
    let (rhs_transpose, _, _, rhs_leading_dimension) = matrix_parameters(rhs)?;
    let mut output = vec![T::zero(); lhs.rows.checked_mul(rhs.cols)?];

    if simd::is_f32::<T>() {
        atlas_blas::gemm_f32(
            Layout::RowMajor,
            lhs_transpose,
            rhs_transpose,
            lhs.rows,
            rhs.cols,
            lhs.cols,
            1.0,
            simd::cast_slice::<T, f32>(&lhs.data[lhs.offset..]),
            lhs_leading_dimension,
            simd::cast_slice::<T, f32>(&rhs.data[rhs.offset..]),
            rhs_leading_dimension,
            0.0,
            simd::cast_mut_slice::<T, f32>(&mut output),
            rhs.cols,
        )
        .ok()?;
        return Some(output);
    }
    if simd::is_f64::<T>() {
        atlas_blas::gemm_f64(
            Layout::RowMajor,
            lhs_transpose,
            rhs_transpose,
            lhs.rows,
            rhs.cols,
            lhs.cols,
            1.0,
            simd::cast_slice::<T, f64>(&lhs.data[lhs.offset..]),
            lhs_leading_dimension,
            simd::cast_slice::<T, f64>(&rhs.data[rhs.offset..]),
            rhs_leading_dimension,
            0.0,
            simd::cast_mut_slice::<T, f64>(&mut output),
            rhs.cols,
        )
        .ok()?;
        return Some(output);
    }

    None
}

fn matrix_parameters<T: Numeric>(
    matrix: MatrixRef<'_, T>,
) -> Option<(Transpose, usize, usize, usize)> {
    if matrix.is_row_major_contiguous() {
        Some((Transpose::None, matrix.rows, matrix.cols, matrix.cols))
    } else if matrix.is_col_major_contiguous() {
        Some((Transpose::Transpose, matrix.cols, matrix.rows, matrix.rows))
    } else {
        None
    }
}
