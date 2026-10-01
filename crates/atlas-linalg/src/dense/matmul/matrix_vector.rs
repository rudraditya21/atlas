use atlas_ndarray::{NDArray, Numeric};

use super::{
    col_major,
    dispatch::{MatmulBackend, MatmulOperation, select_matmul_backend},
    generic, row_major,
};
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    dense::blas,
    internal::dense::{MatrixRef, VectorRef, matrix_ref, vector_ref},
};

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

    let mut data = vec![T::zero(); lhs.rows];
    matmul_matrix_vector_refs_into(lhs, rhs, &mut data);

    Ok(NDArray::from_shape_vec([lhs.rows], data)?)
}

pub(crate) fn matmul_matrix_vector_refs_into<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
    output: &mut [T],
) {
    debug_assert_eq!(output.len(), lhs.rows);
    if lhs.cols == 0 {
        output.fill(T::zero());
        return;
    }

    if select_matmul_backend(MatmulOperation::MatrixVector(lhs, rhs)) == MatmulBackend::Blas {
        assert!(
            blas::gemv_into(lhs, rhs, false, output),
            "BLAS backend selection guarantees supported matrix-vector operands"
        );
        return;
    }

    if lhs.is_row_major_contiguous() && rhs.is_contiguous() {
        row_major::matrix_vector_into(lhs, rhs, output);
    } else if lhs.is_col_major_contiguous() && rhs.is_contiguous() {
        col_major::matrix_vector_into(lhs, rhs, output);
    } else {
        generic::matrix_vector_into(lhs, rhs, output);
    }
}

#[cfg(test)]
pub(super) fn matmul_matrix_vector_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows];
    row_major::matrix_vector_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_matrix_vector_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows];
    col_major::matrix_vector_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_matrix_vector_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); lhs.rows];
    generic::matrix_vector_into(lhs, rhs, &mut output);
    output
}
