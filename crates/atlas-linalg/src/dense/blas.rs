use atlas_ndarray::Numeric;

use crate::internal::{
    dense::{MatrixRef, VectorRef},
    simd,
};

const DOT_MIN_LENGTH: usize = 256;
const GEMV_MIN_WORK: usize = 4_096;
const GEMM_MIN_WORK: usize = 64 * 64 * 64;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

pub(crate) fn dot<T: Numeric>(lhs: &[T], rhs: &[T]) -> Option<T> {
    if !available() || lhs.len() < DOT_MIN_LENGTH || lhs.len() != rhs.len() {
        return None;
    }
    let length = i32::try_from(lhs.len()).ok()?;

    if simd::is_f32::<T>() {
        let value = unsafe {
            ffi::cblas_sdot(
                length,
                simd::cast_slice::<T, f32>(lhs).as_ptr(),
                1,
                simd::cast_slice::<T, f32>(rhs).as_ptr(),
                1,
            )
        };
        return Some(simd::cast_value(value));
    }
    if simd::is_f64::<T>() {
        let value = unsafe {
            ffi::cblas_ddot(
                length,
                simd::cast_slice::<T, f64>(lhs).as_ptr(),
                1,
                simd::cast_slice::<T, f64>(rhs).as_ptr(),
                1,
            )
        };
        return Some(simd::cast_value(value));
    }

    None
}

pub(crate) fn matrix_vector<T: Numeric>(
    matrix: MatrixRef<'_, T>,
    vector: VectorRef<'_, T>,
) -> Option<Vec<T>> {
    if !available()
        || matrix.rows.saturating_mul(matrix.cols) < GEMV_MIN_WORK
        || !vector.is_contiguous()
    {
        return None;
    }

    let (transpose, stored_rows, stored_cols, leading_dimension) = matrix_parameters(matrix)?;
    let m = i32::try_from(stored_rows).ok()?;
    let n = i32::try_from(stored_cols).ok()?;
    let lda = i32::try_from(leading_dimension).ok()?;
    let mut output = vec![T::zero(); matrix.rows];

    if simd::is_f32::<T>() {
        unsafe {
            ffi::cblas_sgemv(
                CBLAS_ROW_MAJOR,
                transpose,
                m,
                n,
                1.0,
                matrix_ptr_f32(matrix),
                lda,
                simd::cast_slice::<T, f32>(vector.contiguous_slice()).as_ptr(),
                1,
                0.0,
                simd::cast_mut_slice::<T, f32>(&mut output).as_mut_ptr(),
                1,
            );
        }
        return Some(output);
    }
    if simd::is_f64::<T>() {
        unsafe {
            ffi::cblas_dgemv(
                CBLAS_ROW_MAJOR,
                transpose,
                m,
                n,
                1.0,
                matrix_ptr_f64(matrix),
                lda,
                simd::cast_slice::<T, f64>(vector.contiguous_slice()).as_ptr(),
                1,
                0.0,
                simd::cast_mut_slice::<T, f64>(&mut output).as_mut_ptr(),
                1,
            );
        }
        return Some(output);
    }

    None
}

pub(crate) fn vector_matrix<T: Numeric>(
    vector: VectorRef<'_, T>,
    matrix: MatrixRef<'_, T>,
) -> Option<Vec<T>> {
    if !available()
        || matrix.rows.saturating_mul(matrix.cols) < GEMV_MIN_WORK
        || !vector.is_contiguous()
    {
        return None;
    }

    let (matrix_transpose, stored_rows, stored_cols, leading_dimension) =
        matrix_parameters(matrix)?;
    let transpose = if matrix_transpose == CBLAS_NO_TRANS { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    let m = i32::try_from(stored_rows).ok()?;
    let n = i32::try_from(stored_cols).ok()?;
    let lda = i32::try_from(leading_dimension).ok()?;
    let mut output = vec![T::zero(); matrix.cols];

    if simd::is_f32::<T>() {
        unsafe {
            ffi::cblas_sgemv(
                CBLAS_ROW_MAJOR,
                transpose,
                m,
                n,
                1.0,
                matrix_ptr_f32(matrix),
                lda,
                simd::cast_slice::<T, f32>(vector.contiguous_slice()).as_ptr(),
                1,
                0.0,
                simd::cast_mut_slice::<T, f32>(&mut output).as_mut_ptr(),
                1,
            );
        }
        return Some(output);
    }
    if simd::is_f64::<T>() {
        unsafe {
            ffi::cblas_dgemv(
                CBLAS_ROW_MAJOR,
                transpose,
                m,
                n,
                1.0,
                matrix_ptr_f64(matrix),
                lda,
                simd::cast_slice::<T, f64>(vector.contiguous_slice()).as_ptr(),
                1,
                0.0,
                simd::cast_mut_slice::<T, f64>(&mut output).as_mut_ptr(),
                1,
            );
        }
        return Some(output);
    }

    None
}

pub(crate) fn matrix_matrix<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Option<Vec<T>> {
    if !available() || lhs.rows.saturating_mul(lhs.cols).saturating_mul(rhs.cols) < GEMM_MIN_WORK {
        return None;
    }

    let (lhs_transpose, _, _, lhs_leading_dimension) = matrix_parameters(lhs)?;
    let (rhs_transpose, _, _, rhs_leading_dimension) = matrix_parameters(rhs)?;
    let rows = i32::try_from(lhs.rows).ok()?;
    let cols = i32::try_from(rhs.cols).ok()?;
    let inner = i32::try_from(lhs.cols).ok()?;
    let lhs_lda = i32::try_from(lhs_leading_dimension).ok()?;
    let rhs_lda = i32::try_from(rhs_leading_dimension).ok()?;
    let output_lda = i32::try_from(rhs.cols).ok()?;
    let mut output = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        unsafe {
            ffi::cblas_sgemm(
                CBLAS_ROW_MAJOR,
                lhs_transpose,
                rhs_transpose,
                rows,
                cols,
                inner,
                1.0,
                matrix_ptr_f32(lhs),
                lhs_lda,
                matrix_ptr_f32(rhs),
                rhs_lda,
                0.0,
                simd::cast_mut_slice::<T, f32>(&mut output).as_mut_ptr(),
                output_lda,
            );
        }
        return Some(output);
    }
    if simd::is_f64::<T>() {
        unsafe {
            ffi::cblas_dgemm(
                CBLAS_ROW_MAJOR,
                lhs_transpose,
                rhs_transpose,
                rows,
                cols,
                inner,
                1.0,
                matrix_ptr_f64(lhs),
                lhs_lda,
                matrix_ptr_f64(rhs),
                rhs_lda,
                0.0,
                simd::cast_mut_slice::<T, f64>(&mut output).as_mut_ptr(),
                output_lda,
            );
        }
        return Some(output);
    }

    None
}

fn available() -> bool {
    cfg!(atlas_blas) && rayon::current_thread_index().is_none()
}

fn matrix_parameters<T: Numeric>(matrix: MatrixRef<'_, T>) -> Option<(i32, usize, usize, usize)> {
    if matrix.is_row_major_contiguous() {
        Some((CBLAS_NO_TRANS, matrix.rows, matrix.cols, matrix.cols))
    } else if matrix.is_col_major_contiguous() {
        Some((CBLAS_TRANS, matrix.cols, matrix.rows, matrix.rows))
    } else {
        None
    }
}

fn matrix_ptr_f32<T: Numeric>(matrix: MatrixRef<'_, T>) -> *const f32 {
    simd::cast_slice::<T, f32>(&matrix.data[matrix.offset..]).as_ptr()
}

fn matrix_ptr_f64<T: Numeric>(matrix: MatrixRef<'_, T>) -> *const f64 {
    simd::cast_slice::<T, f64>(&matrix.data[matrix.offset..]).as_ptr()
}

#[cfg(atlas_blas)]
mod ffi {
    unsafe extern "C" {
        pub(super) fn cblas_sdot(
            n: i32,
            x: *const f32,
            inc_x: i32,
            y: *const f32,
            inc_y: i32,
        ) -> f32;
        pub(super) fn cblas_ddot(
            n: i32,
            x: *const f64,
            inc_x: i32,
            y: *const f64,
            inc_y: i32,
        ) -> f64;
        pub(super) fn cblas_sgemv(
            layout: i32,
            transpose: i32,
            m: i32,
            n: i32,
            alpha: f32,
            matrix: *const f32,
            lda: i32,
            vector: *const f32,
            inc_x: i32,
            beta: f32,
            output: *mut f32,
            inc_y: i32,
        );
        pub(super) fn cblas_dgemv(
            layout: i32,
            transpose: i32,
            m: i32,
            n: i32,
            alpha: f64,
            matrix: *const f64,
            lda: i32,
            vector: *const f64,
            inc_x: i32,
            beta: f64,
            output: *mut f64,
            inc_y: i32,
        );
        pub(super) fn cblas_sgemm(
            layout: i32,
            transpose_a: i32,
            transpose_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            lhs: *const f32,
            lda: i32,
            rhs: *const f32,
            ldb: i32,
            beta: f32,
            output: *mut f32,
            ldc: i32,
        );
        pub(super) fn cblas_dgemm(
            layout: i32,
            transpose_a: i32,
            transpose_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            lhs: *const f64,
            lda: i32,
            rhs: *const f64,
            ldb: i32,
            beta: f64,
            output: *mut f64,
            ldc: i32,
        );
    }
}

#[cfg(not(atlas_blas))]
mod ffi {
    pub(super) unsafe fn cblas_sdot(_: i32, _: *const f32, _: i32, _: *const f32, _: i32) -> f32 {
        unreachable!()
    }

    pub(super) unsafe fn cblas_ddot(_: i32, _: *const f64, _: i32, _: *const f64, _: i32) -> f64 {
        unreachable!()
    }

    macro_rules! unavailable {
        ($name:ident, $ty:ty) => {
            pub(super) unsafe fn $name(
                _: i32,
                _: i32,
                _: i32,
                _: i32,
                _: $ty,
                _: *const $ty,
                _: i32,
                _: *const $ty,
                _: i32,
                _: $ty,
                _: *mut $ty,
                _: i32,
            ) {
                unreachable!()
            }
        };
    }

    unavailable!(cblas_sgemv, f32);
    unavailable!(cblas_dgemv, f64);

    macro_rules! unavailable_gemm {
        ($name:ident, $ty:ty) => {
            pub(super) unsafe fn $name(
                _: i32,
                _: i32,
                _: i32,
                _: i32,
                _: i32,
                _: i32,
                _: $ty,
                _: *const $ty,
                _: i32,
                _: *const $ty,
                _: i32,
                _: $ty,
                _: *mut $ty,
                _: i32,
            ) {
                unreachable!()
            }
        };
    }

    unavailable_gemm!(cblas_sgemm, f32);
    unavailable_gemm!(cblas_dgemm, f64);
}
