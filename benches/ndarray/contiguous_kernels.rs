#[path = "../support/mod.rs"]
mod common;

use atlas_ndarray::NDArray;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn filled_vector(len: usize, value: f64) -> NDArray<f64> {
    NDArray::from_vec(vec![len], vec![value; len]).unwrap()
}

fn filled_square_matrix(side: usize, value: f64) -> NDArray<f64> {
    NDArray::from_vec(vec![side, side], vec![value; side * side]).unwrap()
}

fn padded_square_source(side: usize, value: f64) -> NDArray<f64> {
    NDArray::from_vec(vec![side, side + 1], vec![value; side * (side + 1)]).unwrap()
}

fn bench_contiguous_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/add/contiguous");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        let lhs = filled_vector(size, 1.0);
        let rhs = filled_vector(size, 2.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(lhs.add(black_box(&rhs)).unwrap()))
        });
    }

    group.finish();
}

fn bench_broadcast_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/add/broadcast");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        common::configure_group(&mut group, elements);
        let lhs = filled_square_matrix(side, 1.0);
        let rhs = NDArray::from_vec(vec![1, side], vec![2.0; side]).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(side)),
            &elements,
            |b, _| b.iter(|| black_box(lhs.add(black_box(&rhs)).unwrap())),
        );
    }

    group.finish();
}

fn bench_scalar_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/add/scalar");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        let array = filled_vector(size, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(array.add(black_box(2.0_f64))))
        });
    }

    group.finish();
}

fn bench_contiguous_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/mul/contiguous");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        let lhs = filled_vector(size, 1.0);
        let rhs = filled_vector(size, 2.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(lhs.mul(black_box(&rhs)).unwrap()))
        });
    }

    group.finish();
}

fn bench_broadcast_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/mul/broadcast");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        common::configure_group(&mut group, elements);
        let lhs = filled_square_matrix(side, 1.0);
        let rhs = NDArray::from_vec(vec![1, side], vec![2.0; side]).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(side)),
            &elements,
            |b, _| b.iter(|| black_box(lhs.mul(black_box(&rhs)).unwrap())),
        );
    }

    group.finish();
}

fn bench_scalar_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/mul/scalar");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        let array = filled_vector(size, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(array.mul(black_box(2.0_f64))))
        });
    }

    group.finish();
}

fn bench_masked_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/masked_fill");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        let source = filled_vector(size, 1.0_f64);
        let mask = NDArray::from_vec(vec![size], (0..size).map(|index| index % 3 == 0).collect())
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("allocation_inclusive", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut output = source.clone();
                    output.masked_fill(&mask, 0.0).unwrap();
                    black_box(output)
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("kernel_only", size), &size, |b, _| {
            b.iter_batched_ref(
                || source.clone(),
                |output| {
                    output.masked_fill(&mask, 0.0).unwrap();
                    black_box(output.data()[0])
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_sum_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/reduction/sum");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);

        let contiguous = filled_square_matrix(side, 1.0);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 1.0);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.sum().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_transpose", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.sum().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_slice", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.sum().unwrap()))
        });
    }

    group.finish();
}

fn bench_sum_axis_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/reduction/sum_axis");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);

        let contiguous = filled_square_matrix(side, 1.0);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 1.0);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.sum_axis(black_box(0_usize)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_transpose", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.sum_axis(black_box(0_usize)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_slice", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.sum_axis(black_box(0_usize)).unwrap()))
        });
    }

    group.finish();
}

fn bench_mean_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/reduction/mean");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);

        let contiguous = filled_square_matrix(side, 1.0);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 1.0);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.mean().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_transpose", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.mean().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_slice", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.mean().unwrap()))
        });
    }

    group.finish();
}

fn bench_mean_axis_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/reduction/mean_axis");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);

        let contiguous = filled_square_matrix(side, 1.0);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 1.0);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.mean_axis(black_box(0_usize)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_transpose", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.mean_axis(black_box(0_usize)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("strided_slice", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.mean_axis(black_box(0_usize)).unwrap()))
        });
    }

    group.finish();
}

fn bench_parallel_reduction_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/reduction/sum/parallel_threshold");

    for size in [1 << 19, 1 << 20, 1 << 21] {
        common::configure_group(&mut group, size);
        let array = filled_vector(size, 1.0_f64);

        group.bench_with_input(BenchmarkId::from_parameter(size), &array, |b, input| {
            b.iter(|| black_box(input.sum().unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    ndarray_contiguous_kernels,
    bench_contiguous_add,
    bench_broadcast_add,
    bench_scalar_add,
    bench_contiguous_mul,
    bench_broadcast_mul,
    bench_scalar_mul,
    bench_masked_fill,
    bench_sum_layouts,
    bench_sum_axis_layouts,
    bench_mean_layouts,
    bench_mean_axis_layouts,
    bench_parallel_reduction_thresholds
);
criterion_main!(ndarray_contiguous_kernels);
