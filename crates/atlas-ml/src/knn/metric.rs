use crate::core::row::LogicalRow;
#[cfg(test)]
use crate::{AtlasMlError, AtlasMlResult};

pub(crate) trait DistanceMetric: Sync {
    fn distance_same_dimension(&self, lhs: &[f64], rhs: &[f64]) -> f64;

    fn distance_to_row(&self, lhs: LogicalRow<'_, f64>, rhs: &[f64]) -> f64 {
        if let Some(lhs) = lhs.contiguous_slice() {
            self.distance_same_dimension(lhs, rhs)
        } else {
            let lhs = (0..lhs.len()).map(|index| lhs.value_at(index)).collect::<Vec<_>>();
            self.distance_same_dimension(&lhs, rhs)
        }
    }

    fn axis_distance_lower_bound(&self, _axis_delta: f64) -> Option<f64> {
        None
    }

    fn ball_distance_lower_bound(
        &self,
        _query: &[f64],
        _center: &[f64],
        _radius: f64,
    ) -> Option<f64> {
        None
    }

    #[cfg(test)]
    fn distance(&self, lhs: &[f64], rhs: &[f64]) -> AtlasMlResult<f64> {
        if lhs.len() != rhs.len() {
            return Err(AtlasMlError::ShapeMismatch {
                op: "knn_distance",
                left: vec![lhs.len()],
                right: vec![rhs.len()],
                reason: "feature dimensions must match",
            });
        }

        Ok(self.distance_same_dimension(lhs, rhs))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SquaredEuclideanDistance;

impl DistanceMetric for SquaredEuclideanDistance {
    fn distance_same_dimension(&self, lhs: &[f64], rhs: &[f64]) -> f64 {
        atlas_linalg::squared_euclidean_distance(lhs, rhs)
            .expect("KNN distance operands have matching dimensions")
    }

    fn distance_to_row(&self, lhs: LogicalRow<'_, f64>, rhs: &[f64]) -> f64 {
        if let Some(lhs) = lhs.contiguous_slice() {
            self.distance_same_dimension(lhs, rhs)
        } else {
            distance_to_row_scalar(lhs, rhs)
        }
    }

    fn axis_distance_lower_bound(&self, axis_delta: f64) -> Option<f64> {
        Some(axis_delta * axis_delta)
    }

    fn ball_distance_lower_bound(&self, query: &[f64], center: &[f64], radius: f64) -> Option<f64> {
        let center_distance = self.distance_same_dimension(query, center).sqrt();
        let lower_distance = (center_distance - radius).max(0.0);

        Some(lower_distance * lower_distance)
    }
}

fn distance_to_row_scalar(lhs: LogicalRow<'_, f64>, rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());

    (0..lhs.len())
        .map(|index| {
            let delta = lhs.value_at(index) - rhs[index];
            delta * delta
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{DistanceMetric, SquaredEuclideanDistance};
    use crate::AtlasMlError;

    struct ManhattanDistance;

    impl DistanceMetric for ManhattanDistance {
        fn distance_same_dimension(&self, lhs: &[f64], rhs: &[f64]) -> f64 {
            lhs.iter().zip(rhs).map(|(left, right)| (left - right).abs()).sum()
        }
    }

    #[test]
    fn distance_rejects_different_feature_dimensions() {
        assert_eq!(
            ManhattanDistance.distance(&[1.0, 2.0], &[1.0]).unwrap_err(),
            AtlasMlError::ShapeMismatch {
                op: "knn_distance",
                left: vec![2],
                right: vec![1],
                reason: "feature dimensions must match",
            }
        );
    }

    #[test]
    fn distance_is_deterministic_for_matching_dimensions() {
        let lhs = [1.0, -2.0, 3.0];
        let rhs = [-1.0, 4.0, 3.0];

        assert_eq!(ManhattanDistance.distance(&lhs, &rhs), Ok(8.0));
        assert_eq!(ManhattanDistance.distance(&lhs, &rhs), Ok(8.0));
    }

    #[test]
    fn squared_euclidean_distance_handles_equal_known_and_high_dimensional_points() {
        let metric = SquaredEuclideanDistance;
        let high_dimensional_lhs = vec![1.0_f64; 1_024];
        let high_dimensional_rhs = vec![0.0_f64; 1_024];

        assert_eq!(metric.distance(&[1.0, -2.0], &[1.0, -2.0]), Ok(0.0));
        assert_eq!(metric.distance(&[1.0, 2.0], &[4.0, 6.0]), Ok(25.0));
        assert_eq!(metric.distance(&high_dimensional_lhs, &high_dimensional_rhs), Ok(1_024.0));
    }

    #[test]
    fn squared_euclidean_distance_supports_logical_view_values() {
        use atlas_ndarray::NDArray;

        let lhs = NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let lhs_values = lhs.view().transpose().iter().copied().collect::<Vec<_>>();
        let rhs_values = rhs.view().transpose().iter().copied().collect::<Vec<_>>();

        assert_eq!(SquaredEuclideanDistance.distance(&lhs_values, &rhs_values), Ok(6.0));
    }
}
