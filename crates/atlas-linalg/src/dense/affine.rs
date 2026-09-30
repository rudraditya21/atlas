use atlas_ndarray::{NDArray, Numeric, OperandMetadata};

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult},
    dense::matmul::matmul_matrix_vector_refs,
    internal::dense::{MatrixRef, VectorRef},
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
    let mut scores = matmul_matrix_vector_refs(matrix, coefficients);
    scores.iter_mut().for_each(|score| *score += intercept);

    Ok(NDArray::from_shape_vec([matrix.rows], scores)?)
}
