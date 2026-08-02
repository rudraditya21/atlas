use atlas_ndarray::{NDArray, Numeric};

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::dense::{MatrixRef, VectorRef, dot_contiguous, matrix_ref, vector_ref};
use crate::internal::simd;

pub(super) fn matmul_matrix_vector<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = matrix_ref(lhs);
    let rhs = vector_ref(rhs);

    if lhs.cols != rhs.len {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.rows, lhs.cols],
            right: vec![rhs.len],
            reason: "matrix column count must match vector length",
        });
    }

    let data = if lhs.is_row_major_contiguous() && rhs.is_contiguous() {
        matmul_matrix_vector_row_major(lhs, rhs)
    } else if lhs.is_col_major_contiguous() && rhs.is_contiguous() {
        matmul_matrix_vector_col_major(lhs, rhs)
    } else {
        matmul_matrix_vector_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([lhs.rows], data)?)
}

pub(super) fn matmul_matrix_vector_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let lhs_values = lhs.row_major_region();
    let rhs = rhs.contiguous_slice();
    let mut data = vec![T::zero(); lhs.rows];

    if simd::is_f32::<T>() {
        let lhs_values = simd::cast_slice::<T, f32>(lhs_values);
        let rhs = simd::cast_slice::<T, f32>(rhs);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (output, lhs_row) in data_f32.iter_mut().zip(lhs_values.chunks_exact(lhs.cols)) {
                *output = simd::dot_contiguous_f32(lhs_row, rhs);
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs_values = simd::cast_slice::<T, f64>(lhs_values);
        let rhs = simd::cast_slice::<T, f64>(rhs);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (output, lhs_row) in data_f64.iter_mut().zip(lhs_values.chunks_exact(lhs.cols)) {
                *output = simd::dot_contiguous_f64(lhs_row, rhs);
            }
        }

        return data;
    }

    for (output, lhs_row) in data.iter_mut().zip(lhs_values.chunks_exact(lhs.cols)) {
        *output = dot_contiguous(lhs_row, rhs);
    }

    data
}

pub(super) fn matmul_matrix_vector_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    if simd::is_f32::<T>() {
        let rhs_values = simd::cast_slice::<T, f32>(rhs.contiguous_slice());
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (k, &rhs_value) in rhs_values.iter().enumerate() {
                let lhs_col = simd::cast_slice::<T, f32>(lhs.contiguous_col_slice(k));
                simd::scaled_accumulate_contiguous_f32(data_f32, lhs_col, rhs_value);
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let rhs_values = simd::cast_slice::<T, f64>(rhs.contiguous_slice());
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (k, &rhs_value) in rhs_values.iter().enumerate() {
                let lhs_col = simd::cast_slice::<T, f64>(lhs.contiguous_col_slice(k));
                simd::scaled_accumulate_contiguous_f64(data_f64, lhs_col, rhs_value);
            }
        }

        return data;
    }

    for k in 0..lhs.cols {
        simd::scaled_accumulate_contiguous(&mut data, lhs.contiguous_col_slice(k), rhs.value_at(k));
    }

    data
}

pub(super) fn matmul_matrix_vector_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    for (row, output) in data.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.cols {
            total += lhs.value_at(row, k) * rhs.value_at(k);
        }

        *output = total;
    }

    data
}
