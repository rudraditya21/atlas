use atlas_ndarray::{NDArray, Numeric, checked_element_count};

use super::{matrix_matrix::matmul_matrix_refs, matrix_vector::matmul_matrix_vector_refs};
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
    let mut data = Vec::with_capacity(output_len);

    for batch in 0..*lhs_batches {
        data.extend(matmul_matrix_refs(
            batch_matrix_ref(lhs, batch, *lhs_rows, *lhs_columns),
            batch_matrix_ref(rhs, batch, *rhs_rows, *rhs_columns),
        ));
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
    let mut data = Vec::with_capacity(output_len);

    for batch in 0..*lhs_batches {
        data.extend(matmul_matrix_vector_refs(
            batch_matrix_ref(lhs, batch, *lhs_rows, *lhs_columns),
            batch_vector_ref(rhs, batch, *rhs_length),
        ));
    }

    Ok(NDArray::from_shape_vec(output_shape, data)?)
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
