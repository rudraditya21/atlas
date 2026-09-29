pub(crate) const ELEMENTWISE_CHUNK_LEN: usize = 1 << 16;
const PARALLEL_ELEMENTWISE_THRESHOLD: usize = 1 << 20;

pub(crate) fn should_parallelize_elementwise(len: usize) -> bool {
    len >= PARALLEL_ELEMENTWISE_THRESHOLD && rayon::current_num_threads() > 1
}
