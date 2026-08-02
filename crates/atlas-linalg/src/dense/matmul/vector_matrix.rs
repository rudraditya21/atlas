use atlas_ndarray::{NDArray, Numeric};

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::dense::{MatrixRef, VectorRef, dot_contiguous, matrix_ref, vector_ref};
use crate::internal::simd;

pub(super) fn matmul_vector_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = vector_ref(lhs);
    let rhs = matrix_ref(rhs);

    if lhs.len != rhs.rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.len],
            right: vec![rhs.rows, rhs.cols],
            reason: "vector length must match matrix row count",
        });
    }

    let data = if lhs.is_contiguous() && rhs.is_row_major_contiguous() {
        matmul_vector_matrix_row_major(lhs, rhs)
    } else if lhs.is_contiguous() && rhs.is_col_major_contiguous() {
        matmul_vector_matrix_col_major(lhs, rhs)
    } else {
        matmul_vector_matrix_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([rhs.cols], data)?)
}

pub(super) fn matmul_vector_matrix_row_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let lhs_values = lhs.contiguous_slice();
    let rhs_values = rhs.row_major_region();
    let mut data = vec![T::zero(); rhs.cols];

    if simd::is_f32::<T>() {
        let lhs_values = simd::cast_slice::<T, f32>(lhs_values);
        let rhs_values = simd::cast_slice::<T, f32>(rhs_values);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (&lhs_value, rhs_row) in lhs_values.iter().zip(rhs_values.chunks_exact(rhs.cols)) {
                simd::scaled_accumulate_contiguous_f32(data_f32, rhs_row, lhs_value);
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs_values = simd::cast_slice::<T, f64>(lhs_values);
        let rhs_values = simd::cast_slice::<T, f64>(rhs_values);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (&lhs_value, rhs_row) in lhs_values.iter().zip(rhs_values.chunks_exact(rhs.cols)) {
                simd::scaled_accumulate_contiguous_f64(data_f64, rhs_row, lhs_value);
            }
        }

        return data;
    }

    for (lhs_value, rhs_row) in lhs_values.iter().copied().zip(rhs_values.chunks_exact(rhs.cols)) {
        simd::scaled_accumulate_contiguous(&mut data, rhs_row, lhs_value);
    }

    data
}

pub(super) fn matmul_vector_matrix_col_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let lhs = lhs.contiguous_slice();
    let mut data = vec![T::zero(); rhs.cols];

    if simd::is_f32::<T>() {
        let lhs = simd::cast_slice::<T, f32>(lhs);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (col, output) in data_f32.iter_mut().enumerate() {
                *output = simd::dot_contiguous_f32(
                    lhs,
                    simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                );
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs = simd::cast_slice::<T, f64>(lhs);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (col, output) in data_f64.iter_mut().enumerate() {
                *output = simd::dot_contiguous_f64(
                    lhs,
                    simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                );
            }
        }

        return data;
    }

    for (col, output) in data.iter_mut().enumerate() {
        *output = dot_contiguous(lhs, rhs.contiguous_col_slice(col));
    }

    data
}

pub(super) fn matmul_vector_matrix_generic<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); rhs.cols];

    for (col, output) in data.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.len {
            total += lhs.value_at(k) * rhs.value_at(k, col);
        }

        *output = total;
    }

    data
}
