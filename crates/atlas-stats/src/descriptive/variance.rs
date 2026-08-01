use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::AtlasStatsResult;
use crate::core::StatsOperand;

use super::{collect_values, mean_of};

pub fn variance<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    let values = collect_values(&input.into(), "variance")?;
    let mean = mean_of(&values);

    Ok(values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64)
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
