use atlas_ndarray::{NDArray, Numeric};

use super::{col_major, generic, row_major};
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::dense::{MatrixRef, VectorRef, matrix_ref, vector_ref},
};

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

    let data = matmul_vector_matrix_refs(lhs, rhs);

    Ok(NDArray::from_shape_vec([rhs.cols], data)?)
}

pub(super) fn matmul_vector_matrix_refs<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    if rhs.cols == 0 {
        return Vec::new();
    }
    if lhs.len == 0 {
        return vec![T::zero(); rhs.cols];
    }

    if lhs.is_contiguous() && rhs.is_row_major_contiguous() {
        matmul_vector_matrix_row_major(lhs, rhs)
    } else if lhs.is_contiguous() && rhs.is_col_major_contiguous() {
        matmul_vector_matrix_col_major(lhs, rhs)
    } else {
        matmul_vector_matrix_generic(lhs, rhs)
    }
}

pub(super) fn matmul_vector_matrix_row_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    row_major::vector_matrix(lhs, rhs)
}

pub(super) fn matmul_vector_matrix_col_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    col_major::vector_matrix(lhs, rhs)
}

pub(super) fn matmul_vector_matrix_generic<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    generic::vector_matrix(lhs, rhs)
}
