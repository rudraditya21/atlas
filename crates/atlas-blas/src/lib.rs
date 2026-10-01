//! Typed CPU BLAS bindings used by Atlas linear algebra.

mod provider;

pub use provider::{Provider, Threading, active_provider};

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
}

impl Layout {
    const fn cblas_value(self) -> i32 {
        match self {
            Self::RowMajor => CBLAS_ROW_MAJOR,
            Self::ColumnMajor => CBLAS_COL_MAJOR,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transpose {
    None,
    Transpose,
}

impl Transpose {
    const fn cblas_value(self) -> i32 {
        match self {
            Self::None => CBLAS_NO_TRANS,
            Self::Transpose => CBLAS_TRANS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasError {
    Unavailable,
    DimensionOverflow,
    InvalidStride,
    InvalidLeadingDimension,
    BufferTooSmall,
}

pub type BlasResult<T> = Result<T, BlasError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendConfiguration {
    provider_name: &'static str,
    integer_width: Option<u8>,
    thread_control_available: bool,
}

impl BackendConfiguration {
    pub const fn provider_name(self) -> &'static str {
        self.provider_name
    }

    pub const fn integer_width(self) -> Option<u8> {
        self.integer_width
    }

    pub const fn thread_control_available(self) -> bool {
        self.thread_control_available
    }
}

pub const fn backend_configuration() -> BackendConfiguration {
    match active_provider() {
        Some(provider) => BackendConfiguration {
            provider_name: provider.name(),
            integer_width: Some(32),
            thread_control_available: provider.thread_control_available(),
        },
        None => BackendConfiguration {
            provider_name: "native",
            integer_width: None,
            thread_control_available: false,
        },
    }
}

pub const fn is_available() -> bool {
    active_provider().is_some()
}

pub fn with_threading<R>(threading: Threading, operation: impl FnOnce() -> R) -> R {
    provider::with_threading(threading, operation)
}

macro_rules! typed_blas {
    ($dot:ident, $gemv:ident, $gemm:ident, $ty:ty, $ffi_dot:ident, $ffi_gemv:ident, $ffi_gemm:ident) => {
        pub fn $dot(
            lhs: &[$ty],
            lhs_stride: usize,
            rhs: &[$ty],
            rhs_stride: usize,
            length: usize,
        ) -> BlasResult<$ty> {
            validate_vector(lhs.len(), length, lhs_stride)?;
            validate_vector(rhs.len(), length, rhs_stride)?;
            if length == 0 {
                return Ok(0.0);
            }
            ensure_available()?;

            Ok(unsafe {
                ffi::$ffi_dot(
                    blas_int(length)?,
                    lhs.as_ptr(),
                    blas_int(lhs_stride)?,
                    rhs.as_ptr(),
                    blas_int(rhs_stride)?,
                )
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn $gemv(
            layout: Layout,
            transpose: Transpose,
            rows: usize,
            cols: usize,
            alpha: $ty,
            matrix: &[$ty],
            leading_dimension: usize,
            vector: &[$ty],
            vector_stride: usize,
            beta: $ty,
            output: &mut [$ty],
            output_stride: usize,
        ) -> BlasResult<()> {
            validate_matrix(matrix.len(), layout, rows, cols, leading_dimension)?;
            let (input_length, output_length) = match transpose {
                Transpose::None => (cols, rows),
                Transpose::Transpose => (rows, cols),
            };
            validate_vector(vector.len(), input_length, vector_stride)?;
            validate_vector(output.len(), output_length, output_stride)?;
            if output_length == 0 {
                return Ok(());
            }
            if input_length == 0 {
                scale_vector(output, output_length, output_stride, beta);
                return Ok(());
            }
            ensure_available()?;

            unsafe {
                ffi::$ffi_gemv(
                    layout.cblas_value(),
                    transpose.cblas_value(),
                    blas_int(rows)?,
                    blas_int(cols)?,
                    alpha,
                    matrix.as_ptr(),
                    blas_int(leading_dimension)?,
                    vector.as_ptr(),
                    blas_int(vector_stride)?,
                    beta,
                    output.as_mut_ptr(),
                    blas_int(output_stride)?,
                );
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn $gemm(
            layout: Layout,
            lhs_transpose: Transpose,
            rhs_transpose: Transpose,
            rows: usize,
            cols: usize,
            inner: usize,
            alpha: $ty,
            lhs: &[$ty],
            lhs_leading_dimension: usize,
            rhs: &[$ty],
            rhs_leading_dimension: usize,
            beta: $ty,
            output: &mut [$ty],
            output_leading_dimension: usize,
        ) -> BlasResult<()> {
            let (lhs_rows, lhs_cols) = stored_dimensions(lhs_transpose, rows, inner);
            let (rhs_rows, rhs_cols) = stored_dimensions(rhs_transpose, inner, cols);
            validate_matrix(lhs.len(), layout, lhs_rows, lhs_cols, lhs_leading_dimension)?;
            validate_matrix(rhs.len(), layout, rhs_rows, rhs_cols, rhs_leading_dimension)?;
            validate_matrix(output.len(), layout, rows, cols, output_leading_dimension)?;
            if rows == 0 || cols == 0 {
                return Ok(());
            }
            if inner == 0 {
                scale_matrix(output, layout, rows, cols, output_leading_dimension, beta);
                return Ok(());
            }
            ensure_available()?;

            unsafe {
                ffi::$ffi_gemm(
                    layout.cblas_value(),
                    lhs_transpose.cblas_value(),
                    rhs_transpose.cblas_value(),
                    blas_int(rows)?,
                    blas_int(cols)?,
                    blas_int(inner)?,
                    alpha,
                    lhs.as_ptr(),
                    blas_int(lhs_leading_dimension)?,
                    rhs.as_ptr(),
                    blas_int(rhs_leading_dimension)?,
                    beta,
                    output.as_mut_ptr(),
                    blas_int(output_leading_dimension)?,
                );
            }
            Ok(())
        }
    };
}

typed_blas!(dot_f32, gemv_f32, gemm_f32, f32, cblas_sdot, cblas_sgemv, cblas_sgemm);
typed_blas!(dot_f64, gemv_f64, gemm_f64, f64, cblas_ddot, cblas_dgemv, cblas_dgemm);

fn ensure_available() -> BlasResult<()> {
    is_available().then_some(()).ok_or(BlasError::Unavailable)
}

fn blas_int(value: usize) -> BlasResult<i32> {
    i32::try_from(value).map_err(|_| BlasError::DimensionOverflow)
}

fn validate_vector(buffer_length: usize, length: usize, stride: usize) -> BlasResult<()> {
    if stride == 0 {
        return Err(BlasError::InvalidStride);
    }
    let required = if length == 0 {
        0
    } else {
        (length - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(1))
            .ok_or(BlasError::DimensionOverflow)?
    };
    if buffer_length < required {
        return Err(BlasError::BufferTooSmall);
    }

    Ok(())
}

fn validate_matrix(
    buffer_length: usize,
    layout: Layout,
    rows: usize,
    cols: usize,
    leading_dimension: usize,
) -> BlasResult<()> {
    let (major, minor) = match layout {
        Layout::RowMajor => (rows, cols),
        Layout::ColumnMajor => (cols, rows),
    };
    if leading_dimension < minor.max(1) {
        return Err(BlasError::InvalidLeadingDimension);
    }
    let required = if major == 0 || minor == 0 {
        0
    } else {
        (major - 1)
            .checked_mul(leading_dimension)
            .and_then(|offset| offset.checked_add(minor))
            .ok_or(BlasError::DimensionOverflow)?
    };
    if buffer_length < required {
        return Err(BlasError::BufferTooSmall);
    }

    Ok(())
}

fn stored_dimensions(
    transpose: Transpose,
    operation_rows: usize,
    operation_cols: usize,
) -> (usize, usize) {
    match transpose {
        Transpose::None => (operation_rows, operation_cols),
        Transpose::Transpose => (operation_cols, operation_rows),
    }
}

fn scale_vector<T>(output: &mut [T], length: usize, stride: usize, beta: T)
where
    T: Copy + std::ops::Mul<Output = T>,
{
    for index in 0..length {
        output[index * stride] = output[index * stride] * beta;
    }
}

fn scale_matrix<T>(
    output: &mut [T],
    layout: Layout,
    rows: usize,
    cols: usize,
    leading_dimension: usize,
    beta: T,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    for row in 0..rows {
        for col in 0..cols {
            let index = match layout {
                Layout::RowMajor => row * leading_dimension + col,
                Layout::ColumnMajor => col * leading_dimension + row,
            };
            output[index] = output[index] * beta;
        }
    }
}

#[cfg(atlas_blas)]
mod ffi {
    pub(super) use crate::provider::{
        cblas_ddot, cblas_dgemm, cblas_dgemv, cblas_sdot, cblas_sgemm, cblas_sgemv,
    };
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
