use atlas_ndarray::{NDArray, Numeric};

use super::{col_major, generic, row_major};
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::dense::{MatrixRef, matrix_ref},
};

pub(super) fn matmul_matrix_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = matrix_ref(lhs);
    let rhs = matrix_ref(rhs);

    if lhs.cols != rhs.rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.rows, lhs.cols],
            right: vec![rhs.rows, rhs.cols],
            reason: "left matrix column count must match right matrix row count",
        });
    }

    let data = if lhs.is_row_major_contiguous() && rhs.is_row_major_contiguous() {
        matmul_matrix_matrix_row_major(lhs, rhs)
    } else if lhs.is_col_major_contiguous() && rhs.is_row_major_contiguous() {
        matmul_matrix_matrix_lhs_col_major(lhs, rhs)
    } else if lhs.is_row_major_contiguous() && rhs.is_col_major_contiguous() {
        matmul_matrix_matrix_rhs_col_major(lhs, rhs)
    } else {
        matmul_matrix_matrix_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([lhs.rows, rhs.cols], data)?)
}

pub(super) fn matmul_matrix_matrix_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    row_major::matrix_matrix(lhs, rhs)
}

pub(super) fn matmul_matrix_matrix_lhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    col_major::matrix_matrix_lhs(lhs, rhs)
}

pub(super) fn matmul_matrix_matrix_rhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    col_major::matrix_matrix_rhs(lhs, rhs)
}

pub(super) fn matmul_matrix_matrix_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    generic::matrix_matrix(lhs, rhs)
}
