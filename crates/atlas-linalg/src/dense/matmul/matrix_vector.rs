use atlas_ndarray::{NDArray, Numeric};

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::dense::{MatrixRef, VectorRef, matrix_ref, vector_ref};

use super::{col_major, generic, row_major};

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
    row_major::matrix_vector(lhs, rhs)
}

pub(super) fn matmul_matrix_vector_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    col_major::matrix_vector(lhs, rhs)
}

pub(super) fn matmul_matrix_vector_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    generic::matrix_vector(lhs, rhs)
}
