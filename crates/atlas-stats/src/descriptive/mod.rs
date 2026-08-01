use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsError, AtlasStatsResult, StatsOperand};

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

pub fn stddev<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    I: Into<StatsOperand<'a, T>>,
{
    Ok(variance(input)?.sqrt())
}

pub fn covariance<'a, T, L, R>(lhs: L, rhs: R) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    validate_vector_pair(&lhs, &rhs, "covariance")?;

    let lhs_values = collect_values(&lhs, "covariance")?;
    let rhs_values = collect_values(&rhs, "covariance")?;
    let lhs_mean = mean_of(&lhs_values);
    let rhs_mean = mean_of(&rhs_values);

    let mut total = 0.0_f64;

    for index in 0..lhs_values.len() {
        total += (lhs_values[index] - lhs_mean) * (rhs_values[index] - rhs_mean);
    }

    Ok(total / lhs_values.len() as f64)
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

fn validate_vector_pair<T: Numeric>(
    lhs: &StatsOperand<'_, T>,
    rhs: &StatsOperand<'_, T>,
    op: &'static str,
) -> AtlasStatsResult<()> {
    if lhs.ndim() != 1 {
        return Err(AtlasStatsError::InvalidInputRank {
            op,
            expected: "rank-1 vector",
            rank: lhs.ndim(),
        });
    }

    if rhs.ndim() != 1 {
        return Err(AtlasStatsError::InvalidInputRank {
            op,
            expected: "rank-1 vector",
            rank: rhs.ndim(),
        });
    }

    if lhs.shape()[0] != rhs.shape()[0] {
        return Err(AtlasStatsError::ShapeMismatch {
            op,
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "vector lengths must match",
        });
    }

    Ok(())
}

fn collect_values<T>(operand: &StatsOperand<'_, T>, op: &'static str) -> AtlasStatsResult<Vec<f64>>
where
    T: Numeric + ToPrimitive,
{
    let len = operand.len();

    if len == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }

    let shape = operand.shape();
    let strides = operand.strides();
    let data = operand.data();
    let base_offset = operand.offset();
    let mut values = Vec::with_capacity(len);

    if shape.is_empty() {
        values.push(
            data[base_offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?,
        );

        return Ok(values);
    }

    let mut index = vec![0usize; shape.len()];

    loop {
        let mut offset = base_offset;

        for axis in 0..shape.len() {
            offset += index[axis] * strides[axis];
        }

        values.push(data[offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?);

        if !advance_index(&mut index, shape) {
            break;
        }
    }

    Ok(values)
}

fn advance_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..shape.len()).rev() {
        index[axis] += 1;

        if index[axis] < shape[axis] {
            return true;
        }

        index[axis] = 0;
    }

    false
}

fn mean_of(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, correlation, covariance, stddev, variance};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn variance_and_stddev_use_population_definition() {
        let array = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

        assert_close(variance(&array).unwrap(), 1.25);
        assert_close(stddev(&array).unwrap(), 1.118_033_988_749_895);
    }

    #[test]
    fn variance_and_stddev_work_for_strided_views() {
        let matrix =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let view = matrix.view().transpose();

        assert_close(variance(view).unwrap(), 35.0 / 12.0);
    }

    #[test]
    fn covariance_uses_population_definition() {
        let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

        assert_close(covariance(&lhs, &rhs).unwrap(), 2.5);
    }

    #[test]
    fn correlation_uses_pearson_centered_definition() {
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 4.0]).unwrap();

        assert_close(correlation(&lhs, &rhs).unwrap(), 0.981_980_506_061_965_7);
    }

    #[test]
    fn singleton_population_statistics_reduce_to_zero_except_correlation() {
        let lhs = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();
        let rhs = NDArray::from_shape_vec([1], vec![9.0_f64]).unwrap();

        assert_close(variance(&lhs).unwrap(), 0.0);
        assert_close(stddev(&lhs).unwrap(), 0.0);
        assert_close(covariance(&lhs, &rhs).unwrap(), 0.0);
        assert!(matches!(
            correlation(&lhs, &rhs).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "correlation" }
        ));
    }

    #[test]
    fn covariance_and_correlation_work_for_vector_inputs() {
        let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

        assert_close(correlation(&lhs, &rhs).unwrap(), 1.0);
    }

    #[test]
    fn descriptive_stats_report_expected_validation_errors() {
        let empty = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();
        let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
        let constant = NDArray::from_shape_vec([3], vec![7.0_f64, 7.0, 7.0]).unwrap();

        assert!(matches!(
            variance(&empty).unwrap_err(),
            AtlasStatsError::EmptyInput { op: "variance" }
        ));
        assert!(matches!(
            covariance(&matrix, &matrix).unwrap_err(),
            AtlasStatsError::InvalidInputRank { op: "covariance", .. }
        ));
        assert!(matches!(
            covariance(&lhs, &rhs).unwrap_err(),
            AtlasStatsError::ShapeMismatch { op: "covariance", .. }
        ));
        assert!(matches!(
            correlation(&constant, &lhs).unwrap_err(),
            AtlasStatsError::ZeroVariance { op: "correlation" }
        ));
    }
}
