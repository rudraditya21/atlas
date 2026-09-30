const PARALLEL_INFERENCE_WORK_THRESHOLD: usize = 1 << 18;

pub(crate) fn should_parallelize_inference(task_count: usize, work_items: usize) -> bool {
    let thread_count = rayon::current_num_threads();

    rayon::current_thread_index().is_none()
        && thread_count > 1
        && task_count >= thread_count
        && work_items >= PARALLEL_INFERENCE_WORK_THRESHOLD
}
