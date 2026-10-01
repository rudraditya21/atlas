//! Typed CPU BLAS bindings used by Atlas linear algebra.

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

#[derive(Clone, Copy)]
pub struct Matrix<'a, T> {
    data: &'a [T],
    offset: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<'a, T> Matrix<'a, T> {
    pub fn new(
        data: &'a [T],
        offset: usize,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        let in_bounds = if rows == 0 || cols == 0 {
            offset <= data.len()
        } else {
            let last_row = offset.checked_add((rows - 1).checked_mul(row_stride)?)?;
            last_row
                .checked_add((cols - 1).checked_mul(col_stride)?)
                .is_some_and(|last| last < data.len())
        };

        in_bounds.then_some(Self { data, offset, rows, cols, row_stride, col_stride })
    }

    fn parameters(self) -> Option<(i32, usize, usize, usize)> {
        if self.col_stride == 1 && self.row_stride == self.cols {
            Some((CBLAS_NO_TRANS, self.rows, self.cols, self.cols))
        } else if self.row_stride == 1 && self.col_stride == self.rows {
            Some((CBLAS_TRANS, self.cols, self.rows, self.rows))
        } else {
            None
        }
    }

    fn as_ptr(self) -> *const T {
        self.data[self.offset..].as_ptr()
    }
}

pub const fn is_available() -> bool {
    cfg!(atlas_blas)
}

macro_rules! typed_blas {
    (
        $dot:ident,
        $matrix_vector:ident,
        $vector_matrix:ident,
        $matrix_matrix:ident,
        $ty:ty,
        $ffi_dot:ident,
        $ffi_gemv:ident,
        $ffi_gemm:ident
    ) => {
        pub fn $dot(lhs: &[$ty], rhs: &[$ty]) -> Option<$ty> {
            if !is_available() || lhs.len() != rhs.len() {
                return None;
            }
            let length = i32::try_from(lhs.len()).ok()?;

            Some(unsafe { ffi::$ffi_dot(length, lhs.as_ptr(), 1, rhs.as_ptr(), 1) })
        }

        pub fn $matrix_vector(
            matrix: Matrix<'_, $ty>,
            vector: &[$ty],
            output: &mut [$ty],
        ) -> Option<()> {
            if !is_available() || vector.len() != matrix.cols || output.len() != matrix.rows {
                return None;
            }
            let (transpose, stored_rows, stored_cols, leading_dimension) = matrix.parameters()?;
            let m = i32::try_from(stored_rows).ok()?;
            let n = i32::try_from(stored_cols).ok()?;
            let lda = i32::try_from(leading_dimension).ok()?;
            unsafe {
                ffi::$ffi_gemv(
                    CBLAS_ROW_MAJOR,
                    transpose,
                    m,
                    n,
                    1.0,
                    matrix.as_ptr(),
                    lda,
                    vector.as_ptr(),
                    1,
                    0.0,
                    output.as_mut_ptr(),
                    1,
                );
            }
            Some(())
        }

        pub fn $vector_matrix(
            vector: &[$ty],
            matrix: Matrix<'_, $ty>,
            output: &mut [$ty],
        ) -> Option<()> {
            if !is_available() || vector.len() != matrix.rows || output.len() != matrix.cols {
                return None;
            }
            let (matrix_transpose, stored_rows, stored_cols, leading_dimension) =
                matrix.parameters()?;
            let transpose =
                if matrix_transpose == CBLAS_NO_TRANS { CBLAS_TRANS } else { CBLAS_NO_TRANS };
            let m = i32::try_from(stored_rows).ok()?;
            let n = i32::try_from(stored_cols).ok()?;
            let lda = i32::try_from(leading_dimension).ok()?;
            unsafe {
                ffi::$ffi_gemv(
                    CBLAS_ROW_MAJOR,
                    transpose,
                    m,
                    n,
                    1.0,
                    matrix.as_ptr(),
                    lda,
                    vector.as_ptr(),
                    1,
                    0.0,
                    output.as_mut_ptr(),
                    1,
                );
            }
            Some(())
        }

        pub fn $matrix_matrix(
            lhs: Matrix<'_, $ty>,
            rhs: Matrix<'_, $ty>,
            output: &mut [$ty],
        ) -> Option<()> {
            if !is_available()
                || lhs.cols != rhs.rows
                || output.len() != lhs.rows.checked_mul(rhs.cols)?
            {
                return None;
            }
            let (lhs_transpose, _, _, lhs_leading_dimension) = lhs.parameters()?;
            let (rhs_transpose, _, _, rhs_leading_dimension) = rhs.parameters()?;
            let rows = i32::try_from(lhs.rows).ok()?;
            let cols = i32::try_from(rhs.cols).ok()?;
            let inner = i32::try_from(lhs.cols).ok()?;
            let lhs_lda = i32::try_from(lhs_leading_dimension).ok()?;
            let rhs_lda = i32::try_from(rhs_leading_dimension).ok()?;
            let output_lda = i32::try_from(rhs.cols).ok()?;
            unsafe {
                ffi::$ffi_gemm(
                    CBLAS_ROW_MAJOR,
                    lhs_transpose,
                    rhs_transpose,
                    rows,
                    cols,
                    inner,
                    1.0,
                    lhs.as_ptr(),
                    lhs_lda,
                    rhs.as_ptr(),
                    rhs_lda,
                    0.0,
                    output.as_mut_ptr(),
                    output_lda,
                );
            }
            Some(())
        }
    };
}

typed_blas!(
    dot_f32,
    matrix_vector_f32,
    vector_matrix_f32,
    matrix_matrix_f32,
    f32,
    cblas_sdot,
    cblas_sgemv,
    cblas_sgemm
);
typed_blas!(
    dot_f64,
    matrix_vector_f64,
    vector_matrix_f64,
    matrix_matrix_f64,
    f64,
    cblas_ddot,
    cblas_dgemv,
    cblas_dgemm
);

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
