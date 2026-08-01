use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsResult, StatsOperand};

use super::variance;

pub fn stddev<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    Ok(variance(input)?.sqrt())
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::stddev;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn stddev_uses_population_definition() {
        let array = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

        assert_close(stddev(&array).unwrap(), 1.118_033_988_749_895);
    }

    #[test]
    fn stddev_reduces_singleton_population_to_zero() {
        let array = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();

        assert_close(stddev(&array).unwrap(), 0.0);
    }
}
