use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsResult, StatsOperand, try_for_each_f64, validate_non_empty};

#[derive(Default)]
struct RunningVariance {
    count: usize,
    mean: f64,
    squared_deviations: f64,
}

impl RunningVariance {
    fn add(&mut self, value: f64) {
        // Welford's recurrence avoids the cancellation in sum-of-squares variance formulas.
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.squared_deviations += delta * (value - self.mean);
    }
}

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
    let mut running = RunningVariance::default();

    try_for_each_f64(&input, op, |value| {
        running.add(value);
        Ok(())
    })?;

    Ok(running.squared_deviations / (len - ddof) as f64)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, stddev, variance, variance_ddof};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    fn assert_relative_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= expected.abs() * 1e-12);
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

    #[test]
    fn variance_remains_stable_for_scaled_ill_conditioned_and_view_inputs() {
        let scaled =
            NDArray::from_shape_vec([4], vec![1.0e-100_f64, 2.0e-100, 3.0e-100, 4.0e-100]).unwrap();
        let ill_conditioned = NDArray::from_shape_vec(
            [4],
            vec![1.0e12_f64 + 1.0, 1.0e12 + 2.0, 1.0e12 + 3.0, 1.0e12 + 4.0],
        )
        .unwrap();
        let constant = NDArray::from_shape_vec([4], vec![7.0_f64; 4]).unwrap();
        let source = NDArray::from_shape_vec(
            [2, 3],
            vec![
                1.0e12_f64 + 1.0,
                1.0e12 + 2.0,
                1.0e12 + 3.0,
                1.0e12 + 4.0,
                1.0e12 + 5.0,
                1.0e12 + 6.0,
            ],
        )
        .unwrap();
        let view = source.view().transpose();
        let materialized = view.to_owned();

        assert_relative_close(variance(&scaled).unwrap(), 1.25e-200);
        assert_close(variance(&ill_conditioned).unwrap(), 1.25);
        assert_close(variance_ddof(&ill_conditioned, 1).unwrap(), 5.0 / 3.0);
        assert_eq!(variance(&constant).unwrap(), 0.0);
        assert_eq!(stddev(&constant).unwrap(), 0.0);
        assert_close(variance(view.clone()).unwrap(), variance(&materialized).unwrap());
        assert_close(stddev(view).unwrap(), stddev(&materialized).unwrap());
    }
}
