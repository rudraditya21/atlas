use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsResult, StatsOperand, mean, try_for_each_f64, validate_non_empty};

pub fn variance<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    variance_with_ddof(input.into(), 0, "variance")
}

pub fn variance_ddof<'a, T, I>(input: I, ddof: usize) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    variance_with_ddof(input.into(), ddof, "variance_ddof")
}

pub(crate) fn variance_with_ddof<T>(
    input: StatsOperand<'_, T>,
    ddof: usize,
    op: &'static str,
) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let len = validate_non_empty(&input, op)?;
    if ddof >= len {
        return Err(crate::core::AtlasStatsError::InvalidDegreesOfFreedom { op, ddof, count: len });
    }
    let mean = mean(&input, op)?;
    let mut total = 0.0_f64;

    try_for_each_f64(&input, op, |value| {
        let delta = value - mean;
        total += delta * delta;
        Ok(())
    })?;

    Ok(total / (len - ddof) as f64)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, variance};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn variance_uses_population_definition() {
        let array = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

        assert_close(variance(&array).unwrap(), 1.25);
    }

    #[test]
    fn variance_works_for_strided_views() {
        let matrix =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let view = matrix.view().transpose();

        assert_close(variance(view).unwrap(), 35.0 / 12.0);
    }

    #[test]
    fn variance_reports_empty_input_errors() {
        let empty = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();

        assert!(matches!(
            variance(&empty).unwrap_err(),
            AtlasStatsError::EmptyInput { op: "variance" }
        ));
    }
}
