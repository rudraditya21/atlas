use atlas_ndarray::{AxisIndex, NDArray, Numeric, checked_element_count};
use num_traits::ToPrimitive;

use super::axis::{axis_output_shape, normalize_axis, try_for_each_axis_pair};
use crate::core::{
    AtlasStatsError, AtlasStatsResult, StatsOperand, try_for_each_vector_pair_f64,
    validate_non_empty_vector_pair,
};

#[derive(Default)]
struct RunningCovariance {
    count: usize,
    lhs_mean: f64,
    rhs_mean: f64,
    total: f64,
}

impl RunningCovariance {
    fn add(&mut self, lhs: f64, rhs: f64) {
        self.count += 1;
        let lhs_delta = lhs - self.lhs_mean;
        self.lhs_mean += lhs_delta / self.count as f64;
        let rhs_delta = rhs - self.rhs_mean;
        self.rhs_mean += rhs_delta / self.count as f64;
        self.total += lhs_delta * (rhs - self.rhs_mean);
    }

    fn covariance(self, ddof: usize) -> f64 {
        self.total / (self.count - ddof) as f64
    }
}

pub fn covariance<'a, T, L, R>(lhs: L, rhs: R) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
{
    covariance_with_ddof(lhs.into(), rhs.into(), 0, "covariance")
}

pub fn covariance_ddof<'a, T, L, R>(lhs: L, rhs: R, ddof: usize) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
{
    covariance_with_ddof(lhs.into(), rhs.into(), ddof, "covariance_ddof")
}

/// Returns population covariances after reducing `axis` from identically shaped inputs.
pub fn covariance_axis<'a, T, L, R, A>(lhs: L, rhs: R, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    covariance_axis_impl(lhs.into(), rhs.into(), axis, 0, "covariance_axis")
}

/// Returns covariances normalized by the observation count along `axis` minus `ddof`.
pub fn covariance_axis_ddof<'a, T, L, R, A>(
    lhs: L,
    rhs: R,
    axis: A,
    ddof: usize,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    covariance_axis_impl(lhs.into(), rhs.into(), axis, ddof, "covariance_axis_ddof")
}

fn covariance_axis_impl<T, A>(
    lhs: StatsOperand<'_, T>,
    rhs: StatsOperand<'_, T>,
    axis: A,
    ddof: usize,
    op: &'static str,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    A: AxisIndex,
{
    if lhs.shape() != rhs.shape() {
        return Err(AtlasStatsError::ShapeMismatch {
            op,
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "input shapes must match",
        });
    }

    let shape = lhs.shape();
    let axis = normalize_axis(axis, shape.len())?;
    let output_shape = axis_output_shape(shape, axis, false);
    let output_len = checked_element_count(&output_shape)?;
    if output_len == 0 {
        return NDArray::from_shape_vec(output_shape, Vec::new()).map_err(Into::into);
    }

    let observations = shape[axis];
    if observations == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }
    if ddof >= observations {
        return Err(AtlasStatsError::InvalidDegreesOfFreedom { op, ddof, count: observations });
    }

    let mut covariances = (0..output_len).map(|_| RunningCovariance::default()).collect::<Vec<_>>();

    try_for_each_axis_pair(
        &lhs,
        &rhs,
        axis,
        |lane, lhs_value, rhs_value| -> AtlasStatsResult<()> {
            covariances[lane].add(
                lhs_value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?,
                rhs_value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?,
            );
            Ok(())
        },
    )?;

    NDArray::from_shape_vec(
        output_shape,
        covariances.into_iter().map(|covariance| covariance.covariance(ddof)).collect(),
    )
    .map_err(Into::into)
}

fn covariance_with_ddof<T>(
    lhs: StatsOperand<'_, T>,
    rhs: StatsOperand<'_, T>,
    ddof: usize,
    op: &'static str,
) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let len = validate_non_empty_vector_pair(&lhs, &rhs, op)?;
    if ddof >= len {
        return Err(AtlasStatsError::InvalidDegreesOfFreedom { op, ddof, count: len });
    }

    let mut running = RunningCovariance::default();

    try_for_each_vector_pair_f64(&lhs, &rhs, op, |left, right| {
        running.add(left, right);
        Ok(())
    })?;

    Ok(running.covariance(ddof))
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::{AtlasNdError, NDArray};

    use crate::{AtlasStatsError, covariance, covariance_axis, covariance_axis_ddof};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn covariance_uses_population_definition() {
        let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

        assert_close(covariance(&lhs, &rhs).unwrap(), 2.5);
    }

    #[test]
    fn covariance_supports_singleton_vectors() {
        let lhs = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();
        let rhs = NDArray::from_shape_vec([1], vec![9.0_f64]).unwrap();

        assert_close(covariance(&lhs, &rhs).unwrap(), 0.0);
    }

    #[test]
    fn covariance_reports_vector_validation_errors() {
        let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert!(matches!(
            covariance(&matrix, &matrix).unwrap_err(),
            AtlasStatsError::InvalidInputRank { op: "covariance", .. }
        ));
        assert!(matches!(
            covariance(&lhs, &rhs).unwrap_err(),
            AtlasStatsError::ShapeMismatch { op: "covariance", .. }
        ));
    }

    #[test]
    fn covariance_axis_reduces_the_selected_observation_axis() {
        let lhs = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3, 2], vec![2.0_f64, 1.0, 4.0, 3.0, 6.0, 5.0]).unwrap();

        assert_eq!(covariance_axis(&lhs, &rhs, 0).unwrap().data(), &[8.0 / 3.0; 2]);
        assert_eq!(covariance_axis_ddof(&lhs, &rhs, 0, 1).unwrap().data(), &[4.0; 2]);
    }

    #[test]
    fn covariance_axis_supports_transposed_views() {
        let lhs = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = NDArray::from_shape_vec([3, 2], vec![2.0_f64, 1.0, 4.0, 3.0, 6.0, 5.0]).unwrap();

        assert_eq!(
            covariance_axis(lhs.view().transpose(), rhs.view().transpose(), 1).unwrap().data(),
            &[8.0 / 3.0; 2]
        );
    }

    #[test]
    fn covariance_axis_supports_singleton_lanes_and_validates_ddof() {
        let lhs = NDArray::from_shape_vec([2, 1], vec![1.0_f64, 2.0]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 1], vec![3.0_f64, 4.0]).unwrap();

        assert_eq!(covariance_axis(&lhs, &rhs, 1).unwrap().data(), &[0.0; 2]);
        assert_eq!(
            covariance_axis_ddof(&lhs, &rhs, 1, 1).unwrap_err(),
            AtlasStatsError::InvalidDegreesOfFreedom {
                op: "covariance_axis_ddof",
                ddof: 1,
                count: 1,
            }
        );
    }

    #[test]
    fn covariance_axis_rejects_invalid_axes() {
        let values = NDArray::from_shape_vec([2, 2], vec![1.0_f64; 4]).unwrap();

        assert_eq!(
            covariance_axis(&values, &values, 2).unwrap_err(),
            AtlasStatsError::NdArray(AtlasNdError::InvalidAxis { axis: 2, ndim: 2 })
        );
    }
}
