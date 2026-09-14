use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use super::correlation::variance_total_is_effectively_zero;
use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

/// Returns the weighted mean normalized by the sum of the weights.
pub fn weighted_mean<'a, 'b, T, W, V, Weights>(values: V, weights: Weights) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    W: Numeric + ToPrimitive,
    V: Into<StatsOperand<'a, T>>,
    Weights: Into<StatsOperand<'b, W>>,
{
    let values = values.into();
    let weights = weights.into();
    weighted_mean_impl(&values, &weights, "weighted_mean").map(|(mean, _)| mean)
}

/// Returns the population weighted variance normalized by the sum of the weights.
pub fn weighted_variance<'a, 'b, T, W, V, Weights>(
    values: V,
    weights: Weights,
) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    W: Numeric + ToPrimitive,
    V: Into<StatsOperand<'a, T>>,
    Weights: Into<StatsOperand<'b, W>>,
{
    let values = values.into();
    let weights = weights.into();
    let (mean, total_weight) = weighted_mean_impl(&values, &weights, "weighted_variance")?;
    let mut total = 0.0;

    for (value, weight) in values.iter().zip(weights.iter()) {
        let value = value
            .to_f64()
            .ok_or(AtlasStatsError::NumericConversionFailed { op: "weighted_variance" })?;
        let weight = weight_value(*weight, "weighted_variance")?;
        total += weight * (value - mean).powi(2);
    }

    Ok(total / total_weight)
}

/// Returns the population weighted covariance normalized by the sum of the weights.
pub fn weighted_covariance<'a, 'b, 'c, T, U, W, L, R, Weights>(
    lhs: L,
    rhs: R,
    weights: Weights,
) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    U: Numeric + ToPrimitive,
    W: Numeric + ToPrimitive,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'b, U>>,
    Weights: Into<StatsOperand<'c, W>>,
{
    const OP: &str = "weighted_covariance";
    let lhs = lhs.into();
    let rhs = rhs.into();
    let weights = weights.into();
    validate_weighted_pair(&lhs, &rhs, &weights, OP)?;

    let mut total_weight = 0.0;
    let mut lhs_total = 0.0;
    let mut rhs_total = 0.0;
    for ((left, right), weight) in lhs.iter().zip(rhs.iter()).zip(weights.iter()) {
        let weight = weight_value(*weight, OP)?;
        total_weight += weight;
        lhs_total +=
            weight * left.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
        rhs_total +=
            weight * right.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
    }
    validate_total_weight(total_weight, OP)?;
    let lhs_mean = lhs_total / total_weight;
    let rhs_mean = rhs_total / total_weight;

    let mut covariance = 0.0;
    for ((left, right), weight) in lhs.iter().zip(rhs.iter()).zip(weights.iter()) {
        covariance += weight_value(*weight, OP)?
            * (left.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?
                - lhs_mean)
            * (right.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?
                - rhs_mean);
    }

    Ok(covariance / total_weight)
}

/// Returns the weighted Pearson correlation of two vectors.
pub fn weighted_correlation<'a, 'b, 'c, T, U, W, L, R, Weights>(
    lhs: L,
    rhs: R,
    weights: Weights,
) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    U: Numeric + ToPrimitive,
    W: Numeric + ToPrimitive,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'b, U>>,
    Weights: Into<StatsOperand<'c, W>>,
{
    const OP: &str = "weighted_correlation";
    let lhs = lhs.into();
    let rhs = rhs.into();
    let weights = weights.into();
    validate_weighted_pair(&lhs, &rhs, &weights, OP)?;

    let mut total_weight = 0.0;
    let mut lhs_total = 0.0;
    let mut rhs_total = 0.0;
    for ((left, right), weight) in lhs.iter().zip(rhs.iter()).zip(weights.iter()) {
        let weight = weight_value(*weight, OP)?;
        total_weight += weight;
        lhs_total +=
            weight * left.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
        rhs_total +=
            weight * right.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
    }
    validate_total_weight(total_weight, OP)?;
    let lhs_mean = lhs_total / total_weight;
    let rhs_mean = rhs_total / total_weight;

    let mut covariance_total = 0.0;
    let mut lhs_variance_total = 0.0;
    let mut rhs_variance_total = 0.0;
    let mut lhs_scale = 0.0_f64;
    let mut rhs_scale = 0.0_f64;
    for ((left, right), weight) in lhs.iter().zip(rhs.iter()).zip(weights.iter()) {
        let weight = weight_value(*weight, OP)?;
        let left = left.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
        let right = right.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
        let lhs_delta = left - lhs_mean;
        let rhs_delta = right - rhs_mean;

        covariance_total += weight * lhs_delta * rhs_delta;
        lhs_variance_total += weight * lhs_delta * lhs_delta;
        rhs_variance_total += weight * rhs_delta * rhs_delta;
        if weight > 0.0 {
            lhs_scale = lhs_scale.max(left.abs());
            rhs_scale = rhs_scale.max(right.abs());
        }
    }

    if variance_total_is_effectively_zero(lhs_variance_total, lhs_mean, lhs_scale)
        || variance_total_is_effectively_zero(rhs_variance_total, rhs_mean, rhs_scale)
    {
        return Err(AtlasStatsError::ZeroVariance { op: OP });
    }

    Ok(covariance_total / (lhs_variance_total.sqrt() * rhs_variance_total.sqrt()))
}

fn weighted_mean_impl<T, W>(
    values: &StatsOperand<'_, T>,
    weights: &StatsOperand<'_, W>,
    op: &'static str,
) -> AtlasStatsResult<(f64, f64)>
where
    T: Numeric + ToPrimitive,
    W: Numeric + ToPrimitive,
{
    validate_weighted_pair(values, values, weights, op)?;
    let mut total_weight = 0.0;
    let mut total = 0.0;
    for (value, weight) in values.iter().zip(weights.iter()) {
        let weight = weight_value(*weight, op)?;
        total_weight += weight;
        total += weight * value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
    }
    validate_total_weight(total_weight, op)?;
    Ok((total / total_weight, total_weight))
}

fn validate_weighted_pair<T, U, W>(
    lhs: &StatsOperand<'_, T>,
    rhs: &StatsOperand<'_, U>,
    weights: &StatsOperand<'_, W>,
    op: &'static str,
) -> AtlasStatsResult<()>
where
    T: Numeric,
    U: Numeric,
    W: Numeric,
{
    for operand in [lhs.ndim(), rhs.ndim(), weights.ndim()] {
        if operand != 1 {
            return Err(AtlasStatsError::InvalidInputRank {
                op,
                expected: "rank-1 vector",
                rank: operand,
            });
        }
    }
    if lhs.shape()[0] != rhs.shape()[0] {
        return Err(AtlasStatsError::ShapeMismatch {
            op,
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "vector lengths must match",
        });
    }
    if lhs.shape()[0] != weights.shape()[0] {
        return Err(AtlasStatsError::ShapeMismatch {
            op,
            left: lhs.shape().to_vec(),
            right: weights.shape().to_vec(),
            reason: "value and weight lengths must match",
        });
    }
    if lhs.shape()[0] == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }
    Ok(())
}

fn weight_value<T: ToPrimitive>(weight: T, op: &'static str) -> AtlasStatsResult<f64> {
    let weight = weight.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
    if !weight.is_finite() || weight < 0.0 {
        return Err(AtlasStatsError::InvalidWeights {
            op,
            reason: "weights must be finite and non-negative",
        });
    }
    Ok(weight)
}

fn validate_total_weight(total_weight: f64, op: &'static str) -> AtlasStatsResult<()> {
    if total_weight > 0.0 && total_weight.is_finite() {
        Ok(())
    } else {
        Err(AtlasStatsError::InvalidWeights {
            op,
            reason: "weights must have a positive finite sum",
        })
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, correlation, weighted_correlation};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    #[test]
    fn weighted_correlation_matches_uniform_weight_correlation() {
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3], vec![2.0_f64, 5.0, 4.0]).unwrap();
        let weights = NDArray::from_shape_vec([3], vec![1.0_f64; 3]).unwrap();

        assert_close(
            weighted_correlation(&lhs, &rhs, &weights).unwrap(),
            correlation(&lhs, &rhs).unwrap(),
        );
    }

    #[test]
    fn weighted_correlation_ignores_zero_weight_values() {
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 100.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3], vec![2.0_f64, 4.0, -100.0]).unwrap();
        let weights = NDArray::from_shape_vec([3], vec![1.0_f64, 1.0, 0.0]).unwrap();

        assert_close(weighted_correlation(&lhs, &rhs, &weights).unwrap(), 1.0);
    }

    #[test]
    fn weighted_correlation_rejects_invalid_weights() {
        let values = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let weights = NDArray::from_shape_vec([3], vec![1.0_f64, -1.0, 1.0]).unwrap();

        assert_eq!(
            weighted_correlation(&values, &values, &weights).unwrap_err(),
            AtlasStatsError::InvalidWeights {
                op: "weighted_correlation",
                reason: "weights must be finite and non-negative",
            }
        );
    }

    #[test]
    fn weighted_correlation_supports_views() {
        let lhs_source = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 9.0]).unwrap();
        let rhs_source = NDArray::from_shape_vec([4], vec![0.0_f64, 2.0, 4.0, 1.0]).unwrap();
        let weights_source = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 0.0]).unwrap();

        assert_close(
            weighted_correlation(
                lhs_source.view().slice([1], [2]).unwrap(),
                rhs_source.view().slice([1], [2]).unwrap(),
                weights_source.view().slice([1], [2]).unwrap(),
            )
            .unwrap(),
            1.0,
        );
    }
}
