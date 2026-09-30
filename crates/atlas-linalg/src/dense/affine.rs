use atlas_ndarray::{NDArray, Numeric, OperandMetadata};
use rayon::prelude::*;

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult},
    dense::matmul::{
        MatmulBackend, MatmulOperation, matmul_matrix_vector_refs, select_matmul_backend,
        should_parallelize_matmul,
    },
    internal::dense::{MatrixRef, VectorRef, dot_contiguous},
};

/// Computes one affine score per matrix row.
pub fn affine<T, M>(matrix: &M, coefficients: &[T], intercept: T) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric,
    M: OperandMetadata<T> + ?Sized,
{
    if matrix.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: "affine",
            expected: "a rank-2 matrix",
            rank: matrix.ndim(),
        });
    }
    if matrix.shape()[1] != coefficients.len() {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "affine",
            left: matrix.shape().to_vec(),
            right: vec![coefficients.len()],
            reason: "matrix column count must match coefficient count",
        });
    }

    let matrix = MatrixRef {
        data: matrix.data(),
        offset: matrix.offset(),
        rows: matrix.shape()[0],
        cols: matrix.shape()[1],
        row_stride: matrix.strides()[0],
        col_stride: matrix.strides()[1],
    };
    let coefficients =
        VectorRef { data: coefficients, offset: 0, len: coefficients.len(), stride: 1 };
    let use_blas = select_matmul_backend(MatmulOperation::MatrixVector(matrix, coefficients))
        == MatmulBackend::Blas;
    let scores = if !use_blas && should_parallelize_matmul(matrix.rows, matrix.cols, 1) {
        let mut scores = vec![T::zero(); matrix.rows];
        scores.par_iter_mut().enumerate().for_each(|(row, score)| {
            let product = if matrix.is_row_major_contiguous() {
                dot_contiguous(matrix.contiguous_row_slice(row), coefficients.contiguous_slice())
            } else {
                (0..matrix.cols).fold(T::zero(), |total, column| {
                    total + matrix.value_at(row, column) * coefficients.value_at(column)
                })
            };
            *score = product + intercept;
        });
        scores
    } else {
        let mut scores = matmul_matrix_vector_refs(matrix, coefficients);
        scores.iter_mut().for_each(|score| *score += intercept);
        scores
    };

    Ok(NDArray::from_shape_vec([matrix.rows], scores)?)
}
