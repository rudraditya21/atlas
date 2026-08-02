use crate::{AtlasNdError, AtlasNdResult};

pub(super) const PARALLEL_REDUCTION_THRESHOLD: usize = 1 << 20;
const PARALLEL_REDUCTION_CHUNK_LEN: usize = 1 << 14;
const PARALLEL_REDUCTION_MIN_CHUNKS_PER_THREAD: usize = 2;

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

pub(super) fn ensure_non_empty_reduction(len: usize, op: &'static str) -> AtlasNdResult<()> {
    if len == 0 {
        return Err(AtlasNdError::EmptyReduction { op });
    }

    Ok(())
}

pub(super) fn ensure_non_empty_axis_reduction(
    axis_len: usize,
    op: &'static str,
) -> AtlasNdResult<()> {
    ensure_non_empty_reduction(axis_len, op)
}
