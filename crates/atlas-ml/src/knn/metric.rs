use crate::{AtlasMlError, AtlasMlResult};

pub(crate) trait DistanceMetric {
    fn distance_same_dimension(&self, lhs: &[f64], rhs: &[f64]) -> f64;

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
        lhs.iter()
            .zip(rhs)
            .map(|(left, right)| {
                let delta = left - right;
                delta * delta
            })
            .sum()
    }
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
