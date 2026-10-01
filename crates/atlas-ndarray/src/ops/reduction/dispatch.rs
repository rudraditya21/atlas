use std::cell::RefCell;

use crate::scratch_support::with_thread_local_buffer;

pub(super) const PARALLEL_REDUCTION_THRESHOLD: usize = 1 << 20;
const PARALLEL_REDUCTION_CHUNK_LEN: usize = 1 << 14;
const PARALLEL_REDUCTION_MIN_CHUNKS_PER_THREAD: usize = 2;

thread_local! {
    static REDUCTION_SCRATCH_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static REDUCTION_SCRATCH_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn with_reduction_scratch_f32<R>(
    len: usize,
    operation: impl FnOnce(&mut [f32]) -> R,
) -> R {
    with_thread_local_buffer(&REDUCTION_SCRATCH_F32, len, 0.0, |scratch| operation(scratch))
}

pub(super) fn with_reduction_scratch_f64<R>(
    len: usize,
    operation: impl FnOnce(&mut [f64]) -> R,
) -> R {
    with_thread_local_buffer(&REDUCTION_SCRATCH_F64, len, 0.0, |scratch| operation(scratch))
}

pub(super) const fn parallel_reduction_chunk_len() -> usize {
    PARALLEL_REDUCTION_CHUNK_LEN
}

pub(super) fn should_parallelize_reduction(work_items: usize) -> bool {
    should_parallelize_reduction_for_threads(work_items, rayon::current_num_threads())
}

pub(super) fn should_parallelize_reduction_for_threads(
    work_items: usize,
    thread_count: usize,
) -> bool {
    if thread_count <= 1 || work_items < PARALLEL_REDUCTION_THRESHOLD {
        return false;
    }

    let chunk_count = work_items.div_ceil(PARALLEL_REDUCTION_CHUNK_LEN);

    chunk_count >= thread_count.saturating_mul(PARALLEL_REDUCTION_MIN_CHUNKS_PER_THREAD)
}
