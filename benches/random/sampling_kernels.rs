use atlas_random::{AtlasRng, normal, uniform};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const VECTOR_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];

fn bench_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("random/uniform");

    for size in VECTOR_SIZES {
        group.throughput(Throughput::Elements(size as u64));
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
    let mut group = c.benchmark_group("random/normal");

    for size in VECTOR_SIZES {
        group.throughput(Throughput::Elements(size as u64));
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
