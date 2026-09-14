use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsError, AtlasStatsResult, StatsOperand, mean, try_for_each_f64};

/// Returns population excess kurtosis of all logical values in `input`.
///
/// The result is `m4 / m2² - 3`, where `mk` is the population central moment of order `k`.
/// Constant inputs have undefined kurtosis and return [`AtlasStatsError::ZeroVariance`].
pub fn kurtosis<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    const OP: &str = "kurtosis";

    let input = input.into();
    let mean = mean(&input, OP)?;
    let mut second_moment = 0.0_f64;
    let mut fourth_moment = 0.0_f64;

    try_for_each_f64(&input, OP, |value| {
        let squared_delta = (value - mean).powi(2);
        second_moment += squared_delta;
        fourth_moment += squared_delta * squared_delta;
        Ok(())
    })?;

    if second_moment == 0.0 {
        return Err(AtlasStatsError::ZeroVariance { op: OP });
    }

    Ok(fourth_moment * input.len()? as f64 / second_moment.powi(2) - 3.0)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, kurtosis};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    #[test]
    fn kurtosis_matches_a_known_distribution() {
        let values = NDArray::from_shape_vec([3], vec![-1.0_f64, 0.0, 1.0]).unwrap();

        assert_close(kurtosis(&values).unwrap(), -1.5);
    }

    #[test]
    fn kurtosis_supports_two_distinct_values() {
        let values = NDArray::from_shape_vec([2], vec![0.0_f64, 1.0]).unwrap();

        assert_close(kurtosis(&values).unwrap(), -2.0);
    }

    #[test]
    fn kurtosis_rejects_constant_input() {
        let values = NDArray::from_shape_vec([3], vec![4.0_f64; 3]).unwrap();

        assert_eq!(
            kurtosis(&values).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "kurtosis" }
        );
    }

    #[test]
    fn kurtosis_uses_logical_values_from_views() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![-1.0_f64, 0.0, 1.0, -1.0, 0.0, 1.0]).unwrap();

        assert_close(kurtosis(values.view().transpose()).unwrap(), -1.5);
    }
}
