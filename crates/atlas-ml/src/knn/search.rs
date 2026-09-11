use atlas_ndarray::OperandMetadata;

use super::{metric::DistanceMetric, neighbor::Neighbor, neighbor_set::BoundedNeighborSet};
use crate::{AtlasMlError, AtlasMlResult};

const OP: &str = "brute_force_knn_search";

pub(crate) fn brute_force_search<F, M>(
    training_features: &F,
    query: &[f64],
    k: usize,
    metric: &M,
) -> AtlasMlResult<Vec<Neighbor>>
where
    F: OperandMetadata<f64> + ?Sized,
    M: DistanceMetric + ?Sized,
{
    if training_features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op: OP,
            expected: "a rank-2 [samples, features] matrix",
            rank: training_features.ndim(),
        });
    }

    let sample_count = training_features.shape()[0];
    if k == 0 || k > sample_count {
        return Err(AtlasMlError::InvalidArgument {
            op: OP,
            reason: "k must be between 1 and the number of training samples",
        });
    }

    let feature_count = training_features.shape()[1];
    if query.len() != feature_count {
        return Err(AtlasMlError::ShapeMismatch {
            op: OP,
            left: vec![feature_count],
            right: vec![query.len()],
            reason: "feature dimensions must match",
        });
    }

    let mut neighbors = BoundedNeighborSet::new(k);
    let mut row = vec![0.0; feature_count];
    for sample_index in 0..sample_count {
        copy_row(training_features, sample_index, &mut row);
        neighbors.insert(Neighbor {
            index: sample_index,
            distance: metric.distance_same_dimension(&row, query),
        });
    }

    Ok(neighbors.neighbors().to_vec())
}

fn copy_row<F>(features: &F, row_index: usize, row: &mut [f64])
where
    F: OperandMetadata<f64> + ?Sized,
{
    let row_offset = features.offset() + row_index * features.strides()[0];
    for (feature_index, value) in row.iter_mut().enumerate() {
        *value = features.data()[row_offset + feature_index * features.strides()[1]];
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::brute_force_search;
    use crate::knn::{metric::SquaredEuclideanDistance, neighbor::Neighbor};

    fn neighbor(index: usize, distance: f64) -> Neighbor {
        Neighbor { index, distance }
    }

    #[test]
    fn finds_exact_nearest_neighbors() {
        let training =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0])
                .unwrap();

        assert_eq!(
            brute_force_search(&training, &[1.5, 1.5], 2, &SquaredEuclideanDistance),
            Ok(vec![neighbor(1, 0.5), neighbor(0, 4.5)])
        );
    }

    #[test]
    fn searches_logical_rows_of_transposed_views() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 2.0, 8.0, 0.0, 2.0, 8.0]).unwrap();
        let training = values.view().transpose();

        assert_eq!(
            brute_force_search(&training, &[1.5, 1.5], 2, &SquaredEuclideanDistance),
            Ok(vec![neighbor(1, 0.5), neighbor(0, 4.5)])
        );
    }

    #[test]
    fn resolves_equal_distances_by_training_index() {
        let training =
            NDArray::from_shape_vec([3, 2], vec![-1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0]).unwrap();

        assert_eq!(
            brute_force_search(&training, &[0.0, 0.0], 2, &SquaredEuclideanDistance),
            Ok(vec![neighbor(0, 1.0), neighbor(1, 1.0)])
        );
    }

    #[test]
    fn returns_each_valid_neighbor_count() {
        let training = NDArray::from_shape_vec([3, 1], vec![3.0_f64, 1.0, 2.0]).unwrap();
        let expected = [
            vec![neighbor(1, 1.0)],
            vec![neighbor(1, 1.0), neighbor(2, 4.0)],
            vec![neighbor(1, 1.0), neighbor(2, 4.0), neighbor(0, 9.0)],
        ];

        for (k, neighbors) in (1..=3).zip(expected) {
            assert_eq!(
                brute_force_search(&training, &[0.0], k, &SquaredEuclideanDistance),
                Ok(neighbors)
            );
        }
    }
}
