#[path = "../support/mod.rs"]
mod common;

use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn square_values(side: usize, row_stride: usize) -> NDArray<f64> {
    NDArray::from_shape_vec(
        [side, row_stride],
        (0..side * row_stride).map(|index| index as f64).collect(),
    )
    .unwrap()
}

fn square_mask(side: usize, row_stride: usize) -> NDArray<bool> {
    NDArray::from_shape_vec(
        [side, row_stride],
        (0..side * row_stride).map(|index| index % 3 != 0).collect(),
    )
    .unwrap()
}

fn bench_map_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/view/map");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);
        let contiguous = square_values(side, side);
        let transposed = contiguous.view().transpose();
        let padded = square_values(side, side + 1);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.map(|value| value * 1.5 + 1.0)))
        });
        group.bench_with_input(BenchmarkId::new("transposed", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.map(|value| value * 1.5 + 1.0)))
        });
        group.bench_with_input(BenchmarkId::new("sliced", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.map(|value| value * 1.5 + 1.0)))
        });
    }

    group.finish();
}

fn bench_sum_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/view/reduction/sum");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);
        let contiguous = square_values(side, side);
        let transposed = contiguous.view().transpose();
        let padded = square_values(side, side + 1);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.sum().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("transposed", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.sum().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("sliced", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.sum().unwrap()))
        });
    }

    group.finish();
}

fn bench_selection_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/view/select");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        common::configure_group(&mut group, elements);
        let contiguous = square_values(side, side);
        let contiguous_mask = square_mask(side, side);
        let transposed = contiguous.view().transpose();
        let transposed_mask = contiguous_mask.view().transpose();
        let padded = square_values(side, side + 1);
        let padded_mask = square_mask(side, side + 1);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();
        let sliced_mask = padded_mask.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", &label), &elements, |b, _| {
            b.iter(|| black_box(contiguous.select(&contiguous_mask).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("transposed", &label), &elements, |b, _| {
            b.iter(|| black_box(transposed.select(&transposed_mask).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("sliced", &label), &elements, |b, _| {
            b.iter(|| black_box(sliced.select(&sliced_mask).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    ndarray_view_kernels,
    bench_map_layouts,
    bench_sum_layouts,
    bench_selection_layouts
);
criterion_main!(ndarray_view_kernels);
