use atlas_ndarray::{AxisIndex, NDArray, Numeric, checked_element_count};
use num_traits::ToPrimitive;

use super::axis::normalize_axis;
use crate::core::{
    AtlasStatsError, AtlasStatsResult, StatsOperand, means, try_for_each_vector_pair_f64,
    validate_vector_pair,
};

#[derive(Default)]
struct RunningCorrelation {
    count: usize,
    lhs_mean: f64,
    rhs_mean: f64,
    lhs_variance_total: f64,
    rhs_variance_total: f64,
    covariance_total: f64,
    lhs_scale: f64,
    rhs_scale: f64,
}

impl RunningCorrelation {
    fn add(&mut self, lhs: f64, rhs: f64) {
        self.count += 1;
        let lhs_delta = lhs - self.lhs_mean;
        self.lhs_mean += lhs_delta / self.count as f64;
        let rhs_delta = rhs - self.rhs_mean;
        self.rhs_mean += rhs_delta / self.count as f64;
        self.lhs_variance_total += lhs_delta * (lhs - self.lhs_mean);
        self.rhs_variance_total += rhs_delta * (rhs - self.rhs_mean);
        self.covariance_total += lhs_delta * (rhs - self.rhs_mean);
        self.lhs_scale = self.lhs_scale.max(lhs.abs());
        self.rhs_scale = self.rhs_scale.max(rhs.abs());
    }

    fn correlation(self, op: &'static str) -> AtlasStatsResult<f64> {
        if variance_total_is_effectively_zero(
            self.lhs_variance_total,
            self.lhs_mean,
            self.lhs_scale,
        ) || variance_total_is_effectively_zero(
            self.rhs_variance_total,
            self.rhs_mean,
            self.rhs_scale,
        ) {
            return Err(AtlasStatsError::ZeroVariance { op });
        }

        Ok(self.covariance_total
            / (self.lhs_variance_total.sqrt() * self.rhs_variance_total.sqrt()))
    }
}

pub fn correlation<'a, T, L, R>(lhs: L, rhs: R) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    validate_vector_pair(&lhs, &rhs, "correlation")?;

    let (lhs_mean, rhs_mean, _) = means(&lhs, &rhs, "correlation")?;
    let mut covariance_total = 0.0_f64;
    let mut lhs_variance_total = 0.0_f64;
    let mut rhs_variance_total = 0.0_f64;
    let mut lhs_scale = 0.0_f64;
    let mut rhs_scale = 0.0_f64;

    try_for_each_vector_pair_f64(&lhs, &rhs, "correlation", |left, right| {
        let lhs_delta = left - lhs_mean;
        let rhs_delta = right - rhs_mean;
        lhs_scale = lhs_scale.max(left.abs());
        rhs_scale = rhs_scale.max(right.abs());

        covariance_total += lhs_delta * rhs_delta;
        lhs_variance_total += lhs_delta * lhs_delta;
        rhs_variance_total += rhs_delta * rhs_delta;
        Ok(())
    })?;

    if variance_total_is_effectively_zero(lhs_variance_total, lhs_mean, lhs_scale)
        || variance_total_is_effectively_zero(rhs_variance_total, rhs_mean, rhs_scale)
    {
        return Err(AtlasStatsError::ZeroVariance { op: "correlation" });
    }

    Ok(covariance_total / (lhs_variance_total.sqrt() * rhs_variance_total.sqrt()))
}

/// Returns Pearson correlations after reducing `axis` from identically shaped inputs.
pub fn correlation_axis<'a, T, L, R, A>(lhs: L, rhs: R, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    const OP: &str = "correlation_axis";

    let lhs = lhs.into();
    let rhs = rhs.into();
    if lhs.shape() != rhs.shape() {
        return Err(AtlasStatsError::ShapeMismatch {
            op: OP,
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "input shapes must match",
        });
    }

    let shape = lhs.shape();
    let axis = normalize_axis(axis, shape.len())?;
    let mut output_shape = shape.to_vec();
    output_shape.remove(axis);
    let output_len = checked_element_count(&output_shape)?;
    if output_len == 0 {
        return NDArray::from_shape_vec(output_shape, Vec::new()).map_err(Into::into);
    }

    let observations = shape[axis];
    if observations == 0 {
        return Err(AtlasStatsError::EmptyInput { op: OP });
    }

    let inner_len = shape[axis + 1..].iter().product::<usize>();
    let block_len = observations * inner_len;
    let mut correlations =
        (0..output_len).map(|_| RunningCorrelation::default()).collect::<Vec<_>>();

    for (linear_index, (lhs_value, rhs_value)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let outer = linear_index / block_len;
        let inner = linear_index % inner_len;
        correlations[outer * inner_len + inner].add(
            lhs_value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?,
            rhs_value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?,
        );
    }

    let mut values = Vec::with_capacity(output_len);
    for correlation in correlations {
        values.push(correlation.correlation(OP)?);
    }
    NDArray::from_shape_vec(output_shape, values).map_err(Into::into)
}

pub(super) fn variance_total_is_effectively_zero(total: f64, mean: f64, scale: f64) -> bool {
    let scale = scale.max(mean.abs()).max(f64::MIN_POSITIVE);
    let tolerance = f64::EPSILON.sqrt() * scale;

    total.sqrt() <= tolerance
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, correlation, correlation_axis};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn correlation_uses_pearson_centered_definition() {
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 4.0]).unwrap();

        assert_close(correlation(&lhs, &rhs).unwrap(), 0.981_980_506_061_965_7);
    }

    #[test]
    fn correlation_returns_one_for_perfect_linear_relationships() {
        let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

        assert_close(correlation(&lhs, &rhs).unwrap(), 1.0);
    }

    #[test]
    fn correlation_reports_zero_variance_errors() {
        let constant = NDArray::from_shape_vec([3], vec![7.0_f64, 7.0, 7.0]).unwrap();
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();

        assert!(matches!(
            correlation(&constant, &lhs).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "correlation" }
        ));
    }

    #[test]
    fn correlation_rejects_near_constant_large_magnitude_inputs() {
        let near_constant =
            NDArray::from_shape_vec([3], vec![1.0e16_f64, 1.0e16 + 1.0, 1.0e16 + 2.0]).unwrap();
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();

        assert_eq!(
            correlation(&near_constant, &lhs).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "correlation" }
        );
    }

    #[test]
    fn correlation_axis_returns_one_for_perfect_lanes() {
        let lhs = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs =
            NDArray::from_shape_vec([3, 2], vec![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();

        for value in correlation_axis(&lhs, &rhs, 0).unwrap().data() {
            assert_close(*value, 1.0);
        }
    }

    #[test]
    fn correlation_axis_rejects_constant_lanes() {
        let lhs = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 1.0, 4.0, 1.0, 6.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3, 2], vec![2.0_f64, 4.0, 3.0, 8.0, 4.0, 12.0]).unwrap();

        assert_eq!(
            correlation_axis(&lhs, &rhs, 0).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "correlation_axis" }
        );
    }

    #[test]
    fn correlation_axis_supports_negative_axes_and_views() {
        let lhs = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 3], vec![3.0_f64, 2.0, 1.0, 6.0, 5.0, 4.0]).unwrap();

        for value in correlation_axis(&lhs, &rhs, -1).unwrap().data() {
            assert_close(*value, -1.0);
        }
        for value in
            correlation_axis(lhs.view().transpose(), rhs.view().transpose(), 0).unwrap().data()
        {
            assert_close(*value, -1.0);
        }
    }
}
