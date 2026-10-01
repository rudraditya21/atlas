use std::cell::Cell;

use atlas_ndarray::{NDArray, Numeric};

use super::{
    batched::{
        matmul_batched_matrix_matrix, matmul_batched_matrix_vector, matmul_batched_vector_matrix,
    },
    matrix_matrix::matmul_matrix_matrix,
    matrix_vector::matmul_matrix_vector,
    vector_matrix::matmul_vector_matrix,
};
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::{
        dense::{MatrixRef, VectorRef},
        simd,
    },
};

// Keep medium square matmuls on the serial path unless each worker receives a
// meaningful row slab. The current benchmark suite covers 128x128 as the
// largest baseline square case, so parallel dispatch should not trigger there
// on wider thread pools.
const PARALLEL_MATMUL_WORK_THRESHOLD: usize = 128 * 128 * 128;
const PARALLEL_MATMUL_MIN_ROWS_PER_THREAD: usize = 32;
const BLAS_DOT_MIN_LENGTH: usize = 256;
const BLAS_GEMV_MIN_WORK: usize = 4_096;
const BLAS_GEMM_MIN_WORK: usize = 64 * 64 * 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatmulBackend {
    Blas,
    NativeSimd,
    Scalar,
}

#[derive(Clone, Copy)]
pub(crate) enum MatmulOperation<'a, T: Numeric> {
    Dot(VectorRef<'a, T>, VectorRef<'a, T>),
    MatrixMatrix(MatrixRef<'a, T>, MatrixRef<'a, T>),
    MatrixVector(MatrixRef<'a, T>, VectorRef<'a, T>),
    VectorMatrix(VectorRef<'a, T>, MatrixRef<'a, T>),
}

thread_local! {
    static MATMUL_PARALLELISM_DISABLED: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn dispatch_matmul<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    match (lhs.ndim(), rhs.ndim()) {
        (1, 2) => matmul_vector_matrix(lhs, rhs),
        (2, 1) => matmul_matrix_vector(lhs, rhs),
        (2, 2) => matmul_matrix_matrix(lhs, rhs),
        (2, 3) => matmul_batched_vector_matrix(lhs, rhs),
        (3, 2) => matmul_batched_matrix_vector(lhs, rhs),
        (3, 3) => matmul_batched_matrix_matrix(lhs, rhs),
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "matmul",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

pub(crate) fn select_matmul_backend<T: Numeric>(
    operation: MatmulOperation<'_, T>,
) -> MatmulBackend {
    if blas_supported(operation) {
        MatmulBackend::Blas
    } else if simd_supported(operation) {
        MatmulBackend::NativeSimd
    } else {
        MatmulBackend::Scalar
    }
}

fn blas_supported<T: Numeric>(operation: MatmulOperation<'_, T>) -> bool {
    if !atlas_blas::is_available()
        || rayon::current_thread_index().is_some()
        || !(simd::is_f32::<T>() || simd::is_f64::<T>())
    {
        return false;
    }

    match operation {
        MatmulOperation::Dot(lhs, rhs) => {
            lhs.is_contiguous()
                && rhs.is_contiguous()
                && lhs.len == rhs.len
                && lhs.len >= BLAS_DOT_MIN_LENGTH
                && dimensions_fit_blas(&[lhs.len])
        }
        MatmulOperation::MatrixMatrix(lhs, rhs) => {
            lhs.rows.saturating_mul(lhs.cols).saturating_mul(rhs.cols) >= BLAS_GEMM_MIN_WORK
                && matrix_supported_by_blas(lhs)
                && matrix_supported_by_blas(rhs)
                && dimensions_fit_blas(&[lhs.rows, lhs.cols, rhs.cols])
        }
        MatmulOperation::MatrixVector(matrix, vector) => {
            matrix.rows.saturating_mul(matrix.cols) >= BLAS_GEMV_MIN_WORK
                && vector.is_contiguous()
                && matrix_supported_by_blas(matrix)
        }
        MatmulOperation::VectorMatrix(vector, matrix) => {
            matrix.rows.saturating_mul(matrix.cols) >= BLAS_GEMV_MIN_WORK
                && vector.is_contiguous()
                && matrix_supported_by_blas(matrix)
        }
    }
}

fn simd_supported<T: Numeric>(operation: MatmulOperation<'_, T>) -> bool {
    if !native_simd_available::<T>() {
        return false;
    }

    match operation {
        MatmulOperation::Dot(lhs, rhs) => {
            lhs.is_contiguous() && rhs.is_contiguous() && lhs.len == rhs.len
        }
        MatmulOperation::MatrixMatrix(lhs, rhs) => {
            matrix_is_contiguous(lhs) && matrix_is_contiguous(rhs)
        }
        MatmulOperation::MatrixVector(matrix, vector)
        | MatmulOperation::VectorMatrix(vector, matrix) => {
            matrix_is_contiguous(matrix) && vector.is_contiguous()
        }
    }
}

fn native_simd_available<T: Numeric>() -> bool {
    if !(simd::is_f32::<T>() || simd::is_f64::<T>()) {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    {
        true
    }

    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx")
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

fn matrix_supported_by_blas<T: Numeric>(matrix: MatrixRef<'_, T>) -> bool {
    matrix_is_contiguous(matrix)
        && dimensions_fit_blas(&[
            matrix.rows,
            matrix.cols,
            matrix.row_stride.max(matrix.col_stride),
        ])
}

fn matrix_is_contiguous<T: Numeric>(matrix: MatrixRef<'_, T>) -> bool {
    matrix.is_row_major_contiguous() || matrix.is_col_major_contiguous()
}

fn dimensions_fit_blas(dimensions: &[usize]) -> bool {
    dimensions.iter().all(|&dimension| i32::try_from(dimension).is_ok())
}

pub(crate) fn should_parallelize_matmul(rows: usize, inner: usize, cols: usize) -> bool {
    !MATMUL_PARALLELISM_DISABLED.get()
        && rayon::current_thread_index().is_none()
        && should_parallelize_matmul_for_threads(rows, inner, cols, rayon::current_num_threads())
}

pub(super) fn with_matmul_parallelism_disabled<R>(operation: impl FnOnce() -> R) -> R {
    MATMUL_PARALLELISM_DISABLED.with(|disabled| {
        struct Restore<'a> {
            disabled: &'a Cell<bool>,
            previous: bool,
        }

        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.disabled.set(self.previous);
            }
        }

        let previous = disabled.replace(true);
        let _restore = Restore { disabled, previous };
        operation()
    })
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

    work_items >= PARALLEL_MATMUL_WORK_THRESHOLD
        && rows >= thread_count.saturating_mul(PARALLEL_MATMUL_MIN_ROWS_PER_THREAD)
}
