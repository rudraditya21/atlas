use atlas_linalg::{cholesky, lu, qr};
use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const SQUARE_SIZES: [usize; 3] = [16, 32, 64];
const TALL_SHAPES: [(usize, usize); 3] = [(32, 16), (64, 32), (128, 64)];

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
            data.push(((row * cols + col) as f64 + 1.0) / cols as f64);
        }
    }

    NDArray::from_shape_vec([rows, cols], data).unwrap()
}

fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/lu");

    for size in SQUARE_SIZES {
        let matrix = general_square_matrix(size);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| lu(black_box(&matrix)).unwrap())
        });
    }

    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/qr");

    for (rows, cols) in TALL_SHAPES {
        let matrix = tall_matrix(rows, cols);
        let parameter = format!("{rows}x{cols}");

        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(parameter), &(rows, cols), |b, _| {
            b.iter(|| qr(black_box(&matrix)).unwrap())
        });
    }

    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/cholesky");

    for size in SQUARE_SIZES {
        let matrix = spd_matrix(size);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| cholesky(black_box(&matrix)).unwrap())
        });
    }

    group.finish();
}

criterion_group!(linalg_factorization_kernels, bench_lu, bench_qr, bench_cholesky);
criterion_main!(linalg_factorization_kernels);
