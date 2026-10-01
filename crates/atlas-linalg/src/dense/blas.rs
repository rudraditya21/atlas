use atlas_blas::{Layout, Threading, Transpose};
use atlas_ndarray::Numeric;

use crate::internal::{
    dense::{MatrixRef, VectorRef},
    simd,
};

pub(crate) fn dot_into<T: Numeric>(lhs: &[T], rhs: &[T], output: &mut T) -> bool {
    if simd::is_f32::<T>() {
        let Some(value) = execute(|| {
            atlas_blas::dot_f32(
                simd::cast_slice::<T, f32>(lhs),
                1,
                simd::cast_slice::<T, f32>(rhs),
                1,
                lhs.len(),
            )
        })
        .ok() else {
            return false;
        };
        *output = simd::cast_value(value);
        return true;
    }
    if simd::is_f64::<T>() {
        let Some(value) = execute(|| {
            atlas_blas::dot_f64(
                simd::cast_slice::<T, f64>(lhs),
                1,
                simd::cast_slice::<T, f64>(rhs),
                1,
                lhs.len(),
            )
        })
        .ok() else {
            return false;
        };
        *output = simd::cast_value(value);
        return true;
    }

    false
}

pub(crate) fn gemv_into<T: Numeric>(
    matrix: MatrixRef<'_, T>,
    vector: VectorRef<'_, T>,
    vector_matrix: bool,
    output: &mut [T],
) -> bool {
    let Some((transpose, stored_rows, stored_cols, leading_dimension)) = matrix_parameters(matrix)
    else {
        return false;
    };
    let transpose = if vector_matrix {
        match transpose {
            Transpose::None => Transpose::Transpose,
            Transpose::Transpose => Transpose::None,
        }
    } else {
        transpose
    };

    if simd::is_f32::<T>() {
        return execute(|| {
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
                simd::cast_mut_slice::<T, f32>(output),
                1,
            )
        })
        .is_ok();
    }
    if simd::is_f64::<T>() {
        return execute(|| {
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
                simd::cast_mut_slice::<T, f64>(output),
                1,
            )
        })
        .is_ok();
    }

    false
}

pub(crate) fn gemm_into<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    output: &mut [T],
) -> bool {
    let Some((lhs_transpose, _, _, lhs_leading_dimension)) = matrix_parameters(lhs) else {
        return false;
    };
    let Some((rhs_transpose, _, _, rhs_leading_dimension)) = matrix_parameters(rhs) else {
        return false;
    };

    if simd::is_f32::<T>() {
        return execute(|| {
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
                simd::cast_mut_slice::<T, f32>(output),
                rhs.cols,
            )
        })
        .is_ok();
    }
    if simd::is_f64::<T>() {
        return execute(|| {
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
                simd::cast_mut_slice::<T, f64>(output),
                rhs.cols,
            )
        })
        .is_ok();
    }

    false
}

fn execute<R>(operation: impl FnOnce() -> R) -> R {
    let threading = if rayon::current_thread_index().is_some() {
        Threading::SingleThreaded
    } else {
        Threading::ProviderDefault
    };
    atlas_blas::with_threading(threading, operation)
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
