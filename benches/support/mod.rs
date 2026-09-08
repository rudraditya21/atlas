#![allow(dead_code)]

use std::time::Duration;

use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};

pub(crate) const VECTOR_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];
pub(crate) const SQUARE_MATRIX_SIDES: [usize; 4] = [32, 128, 512, 1024];
pub(crate) const DENSE_MATMUL_SIDES: [usize; 3] = [32, 64, 128];
pub(crate) const DENSE_MATMUL_RECT_SHAPES: [(usize, usize); 3] =
    [(128, 64), (512, 128), (2048, 256)];
pub(crate) const FACTORIZATION_SQUARE_SIZES: [usize; 3] = [16, 32, 64];
pub(crate) const FACTORIZATION_TALL_SHAPES: [(usize, usize); 3] = [(32, 16), (64, 32), (128, 64)];

const LARGE_INPUT_THRESHOLD: usize = 1 << 18;
const DEFAULT_SAMPLE_SIZE: usize = 100;
const LARGE_INPUT_SAMPLE_SIZE: usize = 50;
const DEFAULT_MEASUREMENT_SECS: u64 = 5;
const LARGE_INPUT_MEASUREMENT_SECS: u64 = 10;
const DEFAULT_WARM_UP_SECS: u64 = 3;

pub(crate) fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, work_items: usize) {
    group.throughput(Throughput::Elements(work_items as u64));
    group.warm_up_time(Duration::from_secs(DEFAULT_WARM_UP_SECS));

    if work_items >= LARGE_INPUT_THRESHOLD {
        group.sample_size(LARGE_INPUT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(LARGE_INPUT_MEASUREMENT_SECS));
    } else {
        group.sample_size(DEFAULT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(DEFAULT_MEASUREMENT_SECS));
    }
}

pub(crate) fn square_label(side: usize) -> String {
    format!("{side}x{side}")
}

pub(crate) fn rect_label(rows: usize, cols: usize) -> String {
    format!("{rows}x{cols}")
}
