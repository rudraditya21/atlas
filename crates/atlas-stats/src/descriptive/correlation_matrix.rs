use atlas_ndarray::{NDArray, Numeric};
use num_traits::ToPrimitive;

use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

use super::{
    correlation::variance_total_is_effectively_zero, covariance_matrix::covariance_summary,
};

/// Returns the Pearson correlation matrix for an `[observations, variables]` input matrix.
///
/// Returns `ZeroVariance` when any variable is constant or effectively constant.
pub fn correlation_matrix<'a, T, I>(input: I) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    const OP: &str = "correlation_matrix";

    let summary = covariance_summary(input.into(), OP)?;
    let variables = summary.means.len();

    for index in 0..variables {
        let variance_total =
            summary.covariance.data()[index * variables + index] * summary.observations as f64;
        if variance_total_is_effectively_zero(
            variance_total,
            summary.means[index],
            summary.scales[index],
        ) {
            return Err(AtlasStatsError::ZeroVariance { op: OP });
        }
    }

    let mut correlation = Vec::with_capacity(variables * variables);
    for row in 0..variables {
        let row_variance = summary.covariance.data()[row * variables + row];
        for column in 0..variables {
            let column_variance = summary.covariance.data()[column * variables + column];
            correlation.push(
                summary.covariance.data()[row * variables + column]
                    / (row_variance * column_variance).sqrt(),
            );
        }
    }

    NDArray::from_shape_vec([variables, variables], correlation).map_err(Into::into)
}
