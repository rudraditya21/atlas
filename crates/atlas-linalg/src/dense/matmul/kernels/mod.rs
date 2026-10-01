use atlas_ndarray::Numeric;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2_fma;
#[cfg(target_arch = "aarch64")]
mod neon;

pub(super) struct Tile<'a, T> {
    pub(super) lhs: &'a [T],
    pub(super) rhs: &'a [T],
    pub(super) output: &'a mut [T],
    pub(super) output_stride: usize,
    pub(super) rows: usize,
    pub(super) inner: usize,
    pub(super) cols: usize,
}

pub(super) fn run_f32(tile: Tile<'_, f32>) {
    validate(&tile);

    #[cfg(target_arch = "aarch64")]
    {
        neon::run_f32(tile);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { avx2_fma::run_f32(tile) };
        return;
    }

    scalar::run(tile);
}

pub(super) fn run_f64(tile: Tile<'_, f64>) {
    validate(&tile);

    #[cfg(target_arch = "aarch64")]
    {
        neon::run_f64(tile);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { avx2_fma::run_f64(tile) };
        return;
    }

    scalar::run(tile);
}

pub(super) fn run_scalar<T: Numeric>(tile: Tile<'_, T>) {
    validate(&tile);
    scalar::run(tile);
}

fn validate<T>(tile: &Tile<'_, T>) {
    debug_assert_eq!(tile.lhs.len(), tile.rows * tile.inner);
    debug_assert_eq!(tile.rhs.len(), tile.inner * tile.cols);
    debug_assert!(
        tile.output.len() >= tile.rows.saturating_sub(1) * tile.output_stride + tile.cols
    );
}
