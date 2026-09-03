use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

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
