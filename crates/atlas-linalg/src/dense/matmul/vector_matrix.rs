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

    let mut data = vec![T::zero(); rhs.cols];
    matmul_vector_matrix_refs_into(lhs, rhs, &mut data);

    Ok(NDArray::from_shape_vec([rhs.cols], data)?)
}

pub(super) fn matmul_vector_matrix_refs_into<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    output: &mut [T],
) {
    debug_assert_eq!(output.len(), rhs.cols);
    if output.is_empty() {
        return;
    }
    if lhs.len == 0 {
        output.fill(T::zero());
        return;
    }

    if select_matmul_backend(MatmulOperation::VectorMatrix(lhs, rhs)) == MatmulBackend::Blas {
        assert!(
            blas::gemv_into(rhs, lhs, true, output),
            "BLAS backend selection guarantees supported vector-matrix operands"
        );
        return;
    }

    if lhs.is_contiguous() && rhs.is_row_major_contiguous() {
        row_major::vector_matrix_into(lhs, rhs, output);
    } else if lhs.is_contiguous() && rhs.is_col_major_contiguous() {
        col_major::vector_matrix_into(lhs, rhs, output);
    } else {
        generic::vector_matrix_into(lhs, rhs, output);
    }
}

#[cfg(test)]
pub(super) fn matmul_vector_matrix_row_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); rhs.cols];
    row_major::vector_matrix_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_vector_matrix_col_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); rhs.cols];
    col_major::vector_matrix_into(lhs, rhs, &mut output);
    output
}

#[cfg(test)]
pub(super) fn matmul_vector_matrix_generic<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut output = vec![T::zero(); rhs.cols];
    generic::vector_matrix_into(lhs, rhs, &mut output);
    output
}
