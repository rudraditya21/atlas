# Reduction performance baselines

Run `cargo bench --bench ndarray_contiguous_kernels` on the target machine and retain Criterion's baseline for comparison.

- `ndarray/reduction/sum`: contiguous, transposed, and sliced whole-array sums.
- `ndarray/reduction/sum_axis`: contiguous and strided axis sums.
- `ndarray/reduction/mean` and `mean_axis`: floating stable-accumulation paths.
- `ndarray/reduction/sum/parallel_threshold`: 512 KiB, 1 MiB, and 2 MiB `f64` vectors to catch threshold regressions.

Treat a regression above 10% on the same machine and Criterion baseline as requiring investigation; compare floating reduction results with the documented numerical tolerance rather than bitwise equality.
