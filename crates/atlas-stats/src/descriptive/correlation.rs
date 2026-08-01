use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{
    AtlasStatsError, AtlasStatsResult, StatsOperand, collect_values, mean_of, validate_vector_pair,
};

pub fn correlation<'a, T, L, R>(lhs: L, rhs: R) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    validate_vector_pair(&lhs, &rhs, "correlation")?;

    let lhs_values = collect_values(&lhs, "correlation")?;
    let rhs_values = collect_values(&rhs, "correlation")?;
    let lhs_mean = mean_of(&lhs_values);
    let rhs_mean = mean_of(&rhs_values);

    let mut covariance_total = 0.0_f64;
    let mut lhs_variance_total = 0.0_f64;
    let mut rhs_variance_total = 0.0_f64;

    for index in 0..lhs_values.len() {
        let lhs_delta = lhs_values[index] - lhs_mean;
        let rhs_delta = rhs_values[index] - rhs_mean;

        covariance_total += lhs_delta * rhs_delta;
        lhs_variance_total += lhs_delta * lhs_delta;
        rhs_variance_total += rhs_delta * rhs_delta;
    }

    if lhs_variance_total == 0.0 || rhs_variance_total == 0.0 {
        return Err(AtlasStatsError::ZeroVariance { op: "correlation" });
    }

    Ok(covariance_total / (lhs_variance_total.sqrt() * rhs_variance_total.sqrt()))
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, correlation};

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
}
