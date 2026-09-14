use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsError, AtlasStatsResult, StatsOperand, mean, try_for_each_f64};

/// Returns the population moment skewness of all logical values in `input`.
///
/// Constant inputs have undefined skewness and return [`AtlasStatsError::ZeroVariance`].
pub fn skewness<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    const OP: &str = "skewness";

    let input = input.into();
    let mean = mean(&input, OP)?;
    let mut second_moment = 0.0_f64;
    let mut third_moment = 0.0_f64;

    try_for_each_f64(&input, OP, |value| {
        let delta = value - mean;
        second_moment += delta * delta;
        third_moment += delta * delta * delta;
        Ok(())
    })?;

    if second_moment == 0.0 {
        return Err(AtlasStatsError::ZeroVariance { op: OP });
    }

    Ok(third_moment / second_moment.powf(1.5) * (input.len()? as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, skewness};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    #[test]
    fn skewness_is_zero_for_symmetric_data() {
        let values = NDArray::from_shape_vec([5], vec![-2.0_f64, -1.0, 0.0, 1.0, 2.0]).unwrap();

        assert_close(skewness(&values).unwrap(), 0.0);
    }

    #[test]
    fn skewness_matches_known_skewed_data() {
        let values = NDArray::from_shape_vec([3], vec![0.0_f64, 0.0, 1.0]).unwrap();

        assert_close(skewness(&values).unwrap(), 1.0 / 2.0_f64.sqrt());
    }

    #[test]
    fn skewness_rejects_constant_input() {
        let values = NDArray::from_shape_vec([3], vec![4.0_f64; 3]).unwrap();

        assert_eq!(
            skewness(&values).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "skewness" }
        );
    }

    #[test]
    fn skewness_propagates_non_finite_values() {
        let values = NDArray::from_shape_vec([3], vec![1.0_f64, f64::NAN, 3.0]).unwrap();

        assert!(skewness(&values).unwrap().is_nan());
    }

    #[test]
    fn skewness_uses_logical_values_from_views() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        assert_close(skewness(values.view().transpose()).unwrap(), 1.0 / 2.0_f64.sqrt());
    }
}
