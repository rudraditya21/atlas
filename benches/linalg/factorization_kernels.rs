#[path = "../support/mod.rs"]
mod common;

use atlas_linalg::{cholesky, lu, qr};
use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn general_square_matrix(size: usize) -> NDArray<f64> {
    let mut data = Vec::with_capacity(size * size);

    for row in 0..size {
        for col in 0..size {
            if row == col {
                data.push((size + row + 1) as f64);
            } else {
                data.push((row + col + 1) as f64 / size as f64);
            }
        }
    }

    NDArray::from_shape_vec([size, size], data).unwrap()
}

fn spd_matrix(size: usize) -> NDArray<f64> {
    let mut data = vec![0.0_f64; size * size];

    for row in 0..size {
        for col in 0..size {
            let value = if row == col {
                (size as f64) + (row as f64) + 2.0
            } else {
                ((row.min(col) + 1) as f64) / size as f64
            };
            data[row * size + col] = value;
            data[col * size + row] = value;
        }
    }

    NDArray::from_shape_vec([size, size], data).unwrap()
}

fn tall_matrix(rows: usize, cols: usize) -> NDArray<f64> {
    let mut data = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            let noise =
                ((row * 37 + col * 19) % 17) as f64 / 17.0 / cols as f64 - 0.5 / cols as f64;
            data.push(if row == col { 1.0 + noise } else { noise });
        }
    }

    NDArray::from_shape_vec([rows, cols], data).unwrap()
}

fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/lu/contiguous");

    for size in common::FACTORIZATION_SQUARE_SIZES {
        let matrix = general_square_matrix(size);

        let work_items = size * size;
        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| b.iter(|| lu(black_box(&matrix)).unwrap()),
        );
    }

    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/qr/contiguous");

    for (rows, cols) in common::FACTORIZATION_TALL_SHAPES {
        let matrix = tall_matrix(rows, cols);
        let parameter = common::rect_label(rows, cols);

        common::configure_group(&mut group, rows * cols);
        group.bench_with_input(BenchmarkId::from_parameter(parameter), &(rows, cols), |b, _| {
            b.iter(|| qr(black_box(&matrix)).unwrap())
        });
    }

    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/cholesky/contiguous");

    for size in common::FACTORIZATION_SQUARE_SIZES {
        let matrix = spd_matrix(size);

        let work_items = size * size;
        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| b.iter(|| cholesky(black_box(&matrix)).unwrap()),
        );
    }

    group.finish();
}

fn bench_lu_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/lu/strided_transpose");

    for size in common::FACTORIZATION_SQUARE_SIZES {
        let matrix = general_square_matrix(size);
        let work_items = size * size;

        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| b.iter(|| lu(black_box(matrix.view().transpose())).unwrap()),
        );
    }

    group.finish();
}

fn bench_qr_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/qr/strided_transpose");

    for (rows, cols) in common::FACTORIZATION_TALL_SHAPES {
        let matrix = tall_matrix(cols, rows);
        let parameter = common::rect_label(rows, cols);

        common::configure_group(&mut group, rows * cols);
        group.bench_with_input(BenchmarkId::from_parameter(parameter), &(rows, cols), |b, _| {
            b.iter(|| qr(black_box(matrix.view().transpose())).unwrap())
        });
    }

    group.finish();
}

fn bench_cholesky_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/cholesky/strided_transpose");

    for size in common::FACTORIZATION_SQUARE_SIZES {
        let matrix = spd_matrix(size);
        let work_items = size * size;

        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| b.iter(|| cholesky(black_box(matrix.view().transpose())).unwrap()),
        );
    }

    group.finish();
}

criterion_group!(
    linalg_factorization_kernels,
    bench_lu,
    bench_lu_transposed,
    bench_qr,
    bench_qr_transposed,
    bench_cholesky,
    bench_cholesky_transposed
);
criterion_main!(linalg_factorization_kernels);
