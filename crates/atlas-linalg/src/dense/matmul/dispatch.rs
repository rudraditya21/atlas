use atlas_ndarray::{NDArray, Numeric};

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

use super::{
    matrix_matrix::matmul_matrix_matrix, matrix_vector::matmul_matrix_vector,
    vector_matrix::matmul_vector_matrix, vector_vector::matmul_vector_vector,
};

const PARALLEL_MATMUL_THRESHOLD: usize = 128 * 128 * 128;
const PARALLEL_MATMUL_MIN_ROWS_PER_THREAD: usize = 16;

pub(super) fn dispatch_matmul<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    match (lhs.ndim(), rhs.ndim()) {
        (1, 1) => matmul_vector_vector(lhs, rhs),
        (1, 2) => matmul_vector_matrix(lhs, rhs),
        (2, 1) => matmul_matrix_vector(lhs, rhs),
        (2, 2) => matmul_matrix_matrix(lhs, rhs),
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "matmul",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

pub(super) fn should_parallelize_matmul(rows: usize, inner: usize, cols: usize) -> bool {
    should_parallelize_matmul_for_threads(rows, inner, cols, rayon::current_num_threads())
}

pub(super) fn should_parallelize_matmul_for_threads(
    rows: usize,
    inner: usize,
    cols: usize,
    thread_count: usize,
) -> bool {
    if thread_count <= 1 {
        return false;
    }

    let work_items = rows.saturating_mul(inner).saturating_mul(cols);

    work_items >= PARALLEL_MATMUL_THRESHOLD
        && rows >= thread_count.saturating_mul(PARALLEL_MATMUL_MIN_ROWS_PER_THREAD)
}
