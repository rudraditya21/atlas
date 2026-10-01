use atlas_ndarray::{NDArray, Numeric, checked_element_count};
use rayon::prelude::*;

use super::{
    dispatch::{
        MatmulBackend, MatmulOperation, select_matmul_backend, with_matmul_parallelism_disabled,
    },
    matrix_matrix::matmul_matrix_refs_into,
    matrix_vector::matmul_matrix_vector_refs_into,
    vector_matrix::matmul_vector_matrix_refs_into,
};

const PARALLEL_BATCHED_MATMUL_WORK_THRESHOLD: usize = 1 << 18;
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::dense::{MatrixRef, VectorRef},
};

pub(super) fn matmul_batched_matrix_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let [lhs_batches, lhs_rows, lhs_columns] = lhs.shape() else {
        unreachable!("batched matmul dispatch only receives rank-three operands");
    };
    let [rhs_batches, rhs_rows, rhs_columns] = rhs.shape() else {
        unreachable!("batched matmul dispatch only receives rank-three operands");
    };

    if lhs_batches != rhs_batches {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "batch dimensions must match",
        });
    }
    if lhs_columns != rhs_rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "left matrix column count must match right matrix row count",
        });
    }

    let output_shape = [*lhs_batches, *lhs_rows, *rhs_columns];
    let output_len = checked_element_count(&output_shape)?;
    let mut data = vec![T::zero(); output_len];
    if output_len == 0 {
        return Ok(NDArray::from_shape_vec(output_shape, data)?);
    }
    let batch_output_len = output_len / *lhs_batches;

    let use_blas = select_matmul_backend(MatmulOperation::MatrixMatrix(
        batch_matrix_ref(lhs, 0, *lhs_rows, *lhs_columns),
        batch_matrix_ref(rhs, 0, *rhs_rows, *rhs_columns),
    )) == MatmulBackend::Blas;
    if !use_blas
        && should_parallelize_batched_matmul(*lhs_batches, *lhs_rows, *lhs_columns, *rhs_columns)
    {
        data.par_chunks_mut(batch_output_len).enumerate().for_each(|(batch, output)| {
            with_matmul_parallelism_disabled(|| {
                matmul_matrix_refs_into(
                    batch_matrix_ref(lhs, batch, *lhs_rows, *lhs_columns),
                    batch_matrix_ref(rhs, batch, *rhs_rows, *rhs_columns),
                    output,
                )
            });
        });
    } else {
        for (batch, output) in data.chunks_exact_mut(batch_output_len).enumerate() {
            matmul_matrix_refs_into(
                batch_matrix_ref(lhs, batch, *lhs_rows, *lhs_columns),
                batch_matrix_ref(rhs, batch, *rhs_rows, *rhs_columns),
                output,
            );
        }
    }

    Ok(NDArray::from_shape_vec(output_shape, data)?)
}

pub(super) fn matmul_batched_matrix_vector<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let [lhs_batches, lhs_rows, lhs_columns] = lhs.shape() else {
        unreachable!("batched matmul dispatch only receives rank-three left operands");
    };
    let [rhs_batches, rhs_length] = rhs.shape() else {
        unreachable!("batched matmul dispatch only receives rank-two right operands");
    };

    if lhs_batches != rhs_batches {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "batch dimensions must match",
        });
    }
    if lhs_columns != rhs_length {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "left matrix column count must match vector length",
        });
    }

    let output_shape = [*lhs_batches, *lhs_rows];
    let output_len = checked_element_count(&output_shape)?;
    let mut data = vec![T::zero(); output_len];
    if output_len == 0 {
        return Ok(NDArray::from_shape_vec(output_shape, data)?);
    }
    let batch_output_len = output_len / *lhs_batches;

    let use_blas = select_matmul_backend(MatmulOperation::MatrixVector(
        batch_matrix_ref(lhs, 0, *lhs_rows, *lhs_columns),
        batch_vector_ref(rhs, 0, *rhs_length),
    )) == MatmulBackend::Blas;
    if !use_blas && should_parallelize_batched_matmul(*lhs_batches, *lhs_rows, *lhs_columns, 1) {
        data.par_chunks_mut(batch_output_len).enumerate().for_each(|(batch, output)| {
            with_matmul_parallelism_disabled(|| {
                matmul_matrix_vector_refs_into(
                    batch_matrix_ref(lhs, batch, *lhs_rows, *lhs_columns),
                    batch_vector_ref(rhs, batch, *rhs_length),
                    output,
                )
            });
        });
    } else {
        for (batch, output) in data.chunks_exact_mut(batch_output_len).enumerate() {
            matmul_matrix_vector_refs_into(
                batch_matrix_ref(lhs, batch, *lhs_rows, *lhs_columns),
                batch_vector_ref(rhs, batch, *rhs_length),
                output,
            );
        }
    }

    Ok(NDArray::from_shape_vec(output_shape, data)?)
}

pub(super) fn matmul_batched_vector_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let [lhs_batches, lhs_length] = lhs.shape() else {
        unreachable!("batched matmul dispatch only receives rank-two left operands");
    };
    let [rhs_batches, rhs_rows, rhs_columns] = rhs.shape() else {
        unreachable!("batched matmul dispatch only receives rank-three right operands");
    };

    if lhs_batches != rhs_batches {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "batch dimensions must match",
        });
    }
    if lhs_length != rhs_rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "left vector length must match matrix row count",
        });
    }

    let output_shape = [*lhs_batches, *rhs_columns];
    let output_len = checked_element_count(&output_shape)?;
    let mut data = vec![T::zero(); output_len];
    if output_len == 0 {
        return Ok(NDArray::from_shape_vec(output_shape, data)?);
    }
    let batch_output_len = output_len / *lhs_batches;

    let use_blas = select_matmul_backend(MatmulOperation::VectorMatrix(
        batch_vector_ref(lhs, 0, *lhs_length),
        batch_matrix_ref(rhs, 0, *rhs_rows, *rhs_columns),
    )) == MatmulBackend::Blas;
    if !use_blas && should_parallelize_batched_matmul(*lhs_batches, 1, *lhs_length, *rhs_columns) {
        data.par_chunks_mut(batch_output_len).enumerate().for_each(|(batch, output)| {
            with_matmul_parallelism_disabled(|| {
                matmul_vector_matrix_refs_into(
                    batch_vector_ref(lhs, batch, *lhs_length),
                    batch_matrix_ref(rhs, batch, *rhs_rows, *rhs_columns),
                    output,
                )
            });
        });
    } else {
        for (batch, output) in data.chunks_exact_mut(batch_output_len).enumerate() {
            matmul_vector_matrix_refs_into(
                batch_vector_ref(lhs, batch, *lhs_length),
                batch_matrix_ref(rhs, batch, *rhs_rows, *rhs_columns),
                output,
            );
        }
    }

    Ok(NDArray::from_shape_vec(output_shape, data)?)
}

fn should_parallelize_batched_matmul(
    batches: usize,
    rows: usize,
    inner: usize,
    columns: usize,
) -> bool {
    batches > 1
        && rayon::current_num_threads() > 1
        && batches.saturating_mul(rows).saturating_mul(inner).saturating_mul(columns)
            >= PARALLEL_BATCHED_MATMUL_WORK_THRESHOLD
}

fn batch_matrix_ref<'operand, 'data, T: Numeric>(
    operand: &'operand LinalgOperand<'data, T>,
    batch: usize,
    rows: usize,
    columns: usize,
) -> MatrixRef<'operand, T> {
    MatrixRef {
        data: operand.data(),
        offset: operand.offset() + batch * operand.strides()[0],
        rows,
        cols: columns,
        row_stride: operand.strides()[1],
        col_stride: operand.strides()[2],
    }
}

fn batch_vector_ref<'operand, 'data, T: Numeric>(
    operand: &'operand LinalgOperand<'data, T>,
    batch: usize,
    length: usize,
) -> VectorRef<'operand, T> {
    VectorRef {
        data: operand.data(),
        offset: operand.offset() + batch * operand.strides()[0],
        len: length,
        stride: operand.strides()[1],
    }
}
