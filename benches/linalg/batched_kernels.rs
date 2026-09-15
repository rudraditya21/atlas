#[path = "../support/mod.rs"]
mod common;

use atlas_linalg::{cholesky, lu, matmul, qr};
use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const BATCH_SIZES: [usize; 2] = [8, 32];
const BATCHED_MATRIX_SIDES: [usize; 2] = [32, 64];

fn filled_matrix(rows: usize, columns: usize, value: f64) -> NDArray<f64> {
    NDArray::from_shape_vec([rows, columns], vec![value; rows * columns]).unwrap()
}

fn filled_batched_matrix(batches: usize, rows: usize, columns: usize, value: f64) -> NDArray<f64> {
    NDArray::from_shape_vec([batches, rows, columns], vec![value; batches * rows * columns])
        .unwrap()
}

fn filled_batched_vector(batches: usize, length: usize, value: f64) -> NDArray<f64> {
    NDArray::from_shape_vec([batches, length], vec![value; batches * length]).unwrap()
}

fn general_square_matrix(side: usize) -> NDArray<f64> {
    let mut values = vec![0.0; side * side];
    for row in 0..side {
        for column in 0..side {
            values[row * side + column] = if row == column {
                (side + 1) as f64
            } else {
                (row + column + 1) as f64 / side as f64
            };
        }
    }
    NDArray::from_shape_vec([side, side], values).unwrap()
}

fn spd_matrix(side: usize) -> NDArray<f64> {
    let mut values = vec![0.0; side * side];
    for row in 0..side {
        for column in 0..side {
            values[row * side + column] =
                if row == column { (side + 2) as f64 } else { 1.0 / (row + column + 2) as f64 };
        }
    }
    NDArray::from_shape_vec([side, side], values).unwrap()
}

fn tall_matrix(rows: usize, columns: usize) -> NDArray<f64> {
    NDArray::from_shape_vec(
        [rows, columns],
        (0..rows * columns)
            .map(|index| {
                let row = index / columns;
                let column = index % columns;
                if row % columns == column { 1.0 } else { 0.0 }
            })
            .collect(),
    )
    .unwrap()
}

fn bench_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/batched/matmul/matrix_matrix");

    for side in common::DENSE_MATMUL_SIDES {
        let lhs = filled_matrix(side, side, 1.0);
        let rhs = filled_matrix(side, side, 2.0);
        let work_items = side * side * side;
        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(side)),
            &side,
            |b, _| b.iter(|| black_box(matmul(&lhs, &rhs).unwrap())),
        );
    }

    group.finish();
}

fn bench_batched_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/batched/matmul/batch_matrix_matrix");

    for batches in BATCH_SIZES {
        for side in BATCHED_MATRIX_SIDES {
            let lhs = filled_batched_matrix(batches, side, side, 1.0);
            let rhs = filled_batched_matrix(batches, side, side, 2.0);
            let work_items = batches * side * side * side;
            common::configure_group(&mut group, work_items);
            group.bench_with_input(
                BenchmarkId::new(format!("batch-{batches}"), common::square_label(side)),
                &(batches, side),
                |b, _| b.iter(|| black_box(matmul(&lhs, &rhs).unwrap())),
            );
        }
    }

    group.finish();
}

fn bench_batched_matrix_vector_products(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/batched/matmul/vector_products");

    for batches in BATCH_SIZES {
        for side in BATCHED_MATRIX_SIDES {
            let matrices = filled_batched_matrix(batches, side, side, 1.0);
            let vectors = filled_batched_vector(batches, side, 2.0);
            let work_items = batches * side * side;
            common::configure_group(&mut group, work_items);
            let parameter = format!("batch-{batches}/{}", common::square_label(side));

            group.bench_with_input(BenchmarkId::new("matrix_vector", &parameter), &side, |b, _| {
                b.iter(|| black_box(matmul(&matrices, &vectors).unwrap()))
            });
            group.bench_with_input(BenchmarkId::new("vector_matrix", &parameter), &side, |b, _| {
                b.iter(|| black_box(matmul(&vectors, &matrices).unwrap()))
            });
        }
    }

    group.finish();
}

fn bench_factorizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/batched/factorization");

    for side in common::FACTORIZATION_SQUARE_SIZES {
        let general = general_square_matrix(side);
        let spd = spd_matrix(side);
        let tall = tall_matrix(side * 2, side);
        common::configure_group(&mut group, side * side);
        let parameter = common::square_label(side);

        group.bench_with_input(BenchmarkId::new("lu", &parameter), &side, |b, _| {
            b.iter(|| black_box(lu(&general).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("qr", &parameter), &side, |b, _| {
            b.iter(|| black_box(qr(&tall).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("cholesky", &parameter), &side, |b, _| {
            b.iter(|| black_box(cholesky(&spd).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    linalg_batched_kernels,
    bench_matrix_multiply,
    bench_batched_matrix_multiply,
    bench_batched_matrix_vector_products,
    bench_factorizations
);
criterion_main!(linalg_batched_kernels);
