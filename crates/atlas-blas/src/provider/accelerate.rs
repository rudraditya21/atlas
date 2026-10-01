use super::Threading;

const BLAS_THREADING_SINGLE_THREADED: i32 = 1;

pub(crate) fn with_threading<R>(threading: Threading, operation: impl FnOnce() -> R) -> R {
    if threading == Threading::ProviderDefault {
        return operation();
    }

    struct Restore(Option<i32>);

    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(previous) = self.0 {
                unsafe {
                    BLASSetThreading(previous);
                }
            }
        }
    }

    let previous = unsafe { BLASGetThreading() };
    let changed = unsafe { BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) } == 0;
    let _restore = Restore(changed.then_some(previous));
    operation()
}

unsafe extern "C" {
    fn BLASGetThreading() -> i32;
    fn BLASSetThreading(threading: i32) -> i32;

    pub(crate) fn cblas_sdot(n: i32, x: *const f32, inc_x: i32, y: *const f32, inc_y: i32) -> f32;
    pub(crate) fn cblas_ddot(n: i32, x: *const f64, inc_x: i32, y: *const f64, inc_y: i32) -> f64;
    pub(crate) fn cblas_sgemv(
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
    pub(crate) fn cblas_dgemv(
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
    pub(crate) fn cblas_sgemm(
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
    pub(crate) fn cblas_dgemm(
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
