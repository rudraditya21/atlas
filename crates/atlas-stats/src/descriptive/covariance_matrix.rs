use atlas_ndarray::{NDArray, Numeric, checked_element_count};
use num_traits::ToPrimitive;

use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

/// Returns the population covariance matrix for an `[observations, variables]` input matrix.
pub fn covariance_matrix<'a, T, I>(input: I) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    const OP: &str = "covariance_matrix";

    let input = input.into();
    if input.ndim() != 2 {
        return Err(AtlasStatsError::InvalidInputRank {
            op: OP,
            expected: "rank-2 [observations, variables] matrix",
            rank: input.ndim(),
        });
    }

    let observations = input.shape()[0];
    let variables = input.shape()[1];
    if observations == 0 {
        return Err(AtlasStatsError::EmptyInput { op: OP });
    }

    checked_element_count(&[variables, variables])?;
    if variables == 0 {
        return NDArray::from_shape_vec([0, 0], Vec::new()).map_err(Into::into);
    }

    let mut means = vec![0.0; variables];
    for (index, value) in input.iter().enumerate() {
        means[index % variables] +=
            value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?;
    }
    for mean in &mut means {
        *mean /= observations as f64;
    }

    let mut covariance = vec![0.0; variables * variables];
    let mut centered = vec![0.0; variables];

    for (index, value) in input.iter().enumerate() {
        let variable = index % variables;
        centered[variable] =
            value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: OP })?
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

    NDArray::from_shape_vec([variables, variables], covariance).map_err(Into::into)
}
