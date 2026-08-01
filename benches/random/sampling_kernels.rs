#[path = "../support/mod.rs"]
mod common;

use atlas_random::{AtlasRng, normal, uniform};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("random/uniform/contiguous");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut rng = AtlasRng::seed_from_u64(1_000 + size as u64);

            b.iter(|| {
                uniform([black_box(size)], black_box(0.0_f64), black_box(1.0_f64), &mut rng)
                    .unwrap()
            })
        });
    }

    group.finish();
}

fn bench_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("random/normal/contiguous");

    for size in common::VECTOR_SIZES {
        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut rng = AtlasRng::seed_from_u64(2_000 + size as u64);

            b.iter(|| {
                normal([black_box(size)], black_box(0.0_f64), black_box(1.0_f64), &mut rng).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(random_sampling_kernels, bench_uniform, bench_normal);
criterion_main!(random_sampling_kernels);
