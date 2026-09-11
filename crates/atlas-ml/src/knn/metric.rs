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

#[cfg(test)]
mod tests {
    use super::DistanceMetric;
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
}
