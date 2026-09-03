use atlas_ndarray::{NDArray, Numeric, checked_element_count};
use num_traits::ToPrimitive;

use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

/// Returns the population covariance matrix for an `[observations, variables]` input matrix.
pub fn covariance_matrix<'a, T, I>(input: I) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    covariance_summary(input.into(), "covariance_matrix").map(|summary| summary.covariance)
}

pub(super) struct CovarianceSummary {
    pub(super) covariance: NDArray<f64>,
    pub(super) means: Vec<f64>,
    pub(super) scales: Vec<f64>,
    pub(super) observations: usize,
}

pub(super) fn covariance_summary<T>(
    input: StatsOperand<'_, T>,
    op: &'static str,
) -> AtlasStatsResult<CovarianceSummary>
where
    T: Numeric + ToPrimitive,
{
    if input.ndim() != 2 {
        return Err(AtlasStatsError::InvalidInputRank {
            op,
            expected: "rank-2 [observations, variables] matrix",
            rank: input.ndim(),
        });
    }

    let observations = input.shape()[0];
    let variables = input.shape()[1];
    if observations == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }

    checked_element_count(&[variables, variables])?;
    if variables == 0 {
        return Ok(CovarianceSummary {
            covariance: NDArray::from_shape_vec([0, 0], Vec::new())?,
            means: Vec::new(),
            scales: Vec::new(),
            observations,
        });
    }

    let mut means = vec![0.0; variables];
    let mut scales = vec![0.0_f64; variables];
    for (index, value) in input.iter().enumerate() {
        let variable = index % variables;
        let value = value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
        means[variable] += value;
        scales[variable] = scales[variable].max(value.abs());
    }
    for mean in &mut means {
        *mean /= observations as f64;
    }

    let mut covariance = vec![0.0; variables * variables];
    let mut centered = vec![0.0; variables];

    for (index, value) in input.iter().enumerate() {
        let variable = index % variables;
        centered[variable] =
            value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?
                - means[variable];

        if variable + 1 == variables {
            for row in 0..variables {
                for column in row..variables {
                    covariance[row * variables + column] += centered[row] * centered[column];
                }
            }
        }
    }

    for row in 0..variables {
        for column in row..variables {
            let value = covariance[row * variables + column] / observations as f64;
            covariance[row * variables + column] = value;
            covariance[column * variables + row] = value;
        }
    }

    Ok(CovarianceSummary {
        covariance: NDArray::from_shape_vec([variables, variables], covariance)?,
        means,
        scales,
        observations,
    })
}
