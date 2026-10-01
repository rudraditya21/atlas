use atlas_ndarray::{NDArray, Numeric};

use super::{
    col_major,
    dispatch::{MatmulBackend, MatmulOperation, select_matmul_backend},
    generic, row_major,
};
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    dense::blas,
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

    let mut data = vec![T::zero(); lhs.rows * rhs.cols];
    matmul_matrix_refs_into(lhs, rhs, &mut data);

    Ok(NDArray::from_shape_vec([lhs.rows, rhs.cols], data)?)
}

pub(super) fn matmul_matrix_refs_into<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    output: &mut [T],
) {
    debug_assert_eq!(output.len(), lhs.rows * rhs.cols);
    if output.is_empty() {
        return;
    }
    if lhs.cols == 0 {
        output.fill(T::zero());
        return;
    }

    if select_matmul_backend(MatmulOperation::MatrixMatrix(lhs, rhs)) == MatmulBackend::Blas {
        assert!(
            blas::gemm_into(lhs, rhs, output),
            "BLAS backend selection guarantees supported matrix operands"
        );
        return;
    }

    if lhs.is_row_major_contiguous() && rhs.is_row_major_contiguous() {
        row_major::matrix_matrix_into(lhs, rhs, output);
    } else if lhs.is_col_major_contiguous() && rhs.is_row_major_contiguous() {
        col_major::matrix_matrix_lhs_into(lhs, rhs, output);
    } else if lhs.is_row_major_contiguous() && rhs.is_col_major_contiguous() {
        col_major::matrix_matrix_rhs_into(lhs, rhs, output);
    } else {
        generic::matrix_matrix_into(lhs, rhs, output);
    }
}

#[cfg(test)]
pub(super) fn matmul_matrix_matrix_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows * rhs.cols];
    row_major::matrix_matrix_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_matrix_matrix_lhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows * rhs.cols];
    col_major::matrix_matrix_lhs_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_matrix_matrix_rhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows * rhs.cols];
    col_major::matrix_matrix_rhs_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_matrix_matrix_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows * rhs.cols];
    generic::matrix_matrix_into(lhs, rhs, &mut output);
    output
}
