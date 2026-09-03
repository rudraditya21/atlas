use num_traits::{Float, ToPrimitive};

use crate::{AtlasNdError, AtlasNdResult, Numeric, internal::value_iter};

use super::metadata::WholeReductionMetadata;

pub(super) fn nanmin_all<T: Numeric + Float>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<T> {
    nan_extreme(data, offset, shape, strides, "nanmin", |value, current| value < current)
}

pub(super) fn nanmax_all<T: Numeric + Float>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<T> {
    nan_extreme(data, offset, shape, strides, "nanmax", |value, current| value > current)
}

pub(super) fn nanmean_all<T: Numeric + Float + ToPrimitive>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<f64> {
    Ok(nan_moments(data, offset, shape, strides, "nanmean")?.mean)
}

pub(super) fn nanstd_all<T: Numeric + Float + ToPrimitive>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<f64> {
    let moments = nan_moments(data, offset, shape, strides, "nanstd")?;
    Ok((moments.sum_squares / moments.count as f64).sqrt())
}

fn nan_extreme<T: Numeric + Float>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    op: &'static str,
    replaces: impl Fn(T, T) -> bool,
) -> AtlasNdResult<T> {
    WholeReductionMetadata::from_shape(shape).require_non_empty(op)?;
    value_iter(data, offset, shape, strides)
        .copied()
        .filter(|value| !value.is_nan())
        .reduce(|current, value| if replaces(value, current) { value } else { current })
        .ok_or(AtlasNdError::AllNaN { op })
}

#[derive(Default)]
struct NanMoments {
    count: usize,
    mean: f64,
    sum_squares: f64,
}

impl NanMoments {
    fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.sum_squares += delta * (value - self.mean);
    }
}

fn nan_moments<T: Numeric + Float + ToPrimitive>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    op: &'static str,
) -> AtlasNdResult<NanMoments> {
    WholeReductionMetadata::from_shape(shape).require_non_empty(op)?;
    let mut moments = NanMoments::default();

    for value in value_iter(data, offset, shape, strides).copied() {
        if !value.is_nan() {
            moments.add(value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?);
        }
    }

    if moments.count == 0 { Err(AtlasNdError::AllNaN { op }) } else { Ok(moments) }
}
