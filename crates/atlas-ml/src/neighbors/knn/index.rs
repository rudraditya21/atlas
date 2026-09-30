use atlas_ndarray::{NDArray, OperandMetadata};

use super::{backend::NeighborSearchBackend, config::KnnSearchAlgorithm, neighbor::Neighbor};
use crate::{AtlasMlError, AtlasMlResult};

pub(crate) struct TrainingIndex {
    features: NDArray<f64>,
    backend: NeighborSearchBackend,
}

impl TrainingIndex {
    pub(crate) fn new(
        features: NDArray<f64>,
        algorithm: KnnSearchAlgorithm,
        tree_leaf_size: usize,
    ) -> AtlasMlResult<Self> {
        let backend = NeighborSearchBackend::new(&features, algorithm, tree_leaf_size)?;
        Ok(Self { features, backend })
    }

    pub(crate) fn features(&self) -> &NDArray<f64> {
        &self.features
    }

    pub(crate) fn search_algorithm(&self) -> KnnSearchAlgorithm {
        self.backend.algorithm()
    }

    pub(crate) fn search(&self, query: &[f64], k: usize) -> AtlasMlResult<Vec<Neighbor>> {
        self.validate_request(query.len(), k)?;
        self.backend.search(&self.features, query, k)
    }

    pub(crate) fn search_batch<Q>(&self, queries: &Q, k: usize) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + Sync + ?Sized,
    {
        if queries.ndim() != 2 {
            return Err(AtlasMlError::InvalidInputRank {
                op: self.backend.search_op(),
                expected: "a rank-2 [samples, features] matrix",
                rank: queries.ndim(),
            });
        }
        self.validate_request(queries.shape()[1], k)?;
        self.backend.search_batch(&self.features, queries, k)
    }

    fn validate_request(&self, query_feature_count: usize, k: usize) -> AtlasMlResult<()> {
        if k == 0 || k > self.features.shape()[0] {
            return Err(AtlasMlError::InvalidArgument {
                op: self.backend.search_op(),
                reason: "k must be between 1 and the number of training samples",
            });
        }
        if query_feature_count != self.features.shape()[1] {
            return Err(AtlasMlError::ShapeMismatch {
                op: self.backend.search_op(),
                left: vec![self.features.shape()[1]],
                right: vec![query_feature_count],
                reason: "feature dimensions must match",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::TrainingIndex;
    use crate::neighbors::knn::{config::KnnSearchAlgorithm, neighbor::Neighbor};

    #[test]
    fn retains_owned_training_features_and_shape() {
        let index = TrainingIndex::new(
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            KnnSearchAlgorithm::BruteForce,
            1,
        )
        .unwrap();

        assert_eq!(index.features().shape(), &[2, 3]);
        assert_eq!(index.features().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn initializes_the_requested_search_backend() {
        let index = TrainingIndex::new(
            NDArray::from_shape_vec([2, 1], vec![0.0_f64, 2.0]).unwrap(),
            KnnSearchAlgorithm::BruteForce,
            1,
        )
        .unwrap();

        assert_eq!(index.search_algorithm(), KnnSearchAlgorithm::BruteForce);
        assert_eq!(index.search(&[1.0], 1), Ok(vec![Neighbor { index: 0, distance: 1.0 }]));
    }

    #[test]
    fn backends_preserve_batch_ties_views_empty_results_and_errors() {
        let query_source =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 0.5, 99.0, 0.0, 0.0, 99.0]).unwrap();
        let queries = query_source.view().transpose().slice([0, 0], [2, 2]).unwrap();
        let empty_queries = NDArray::<f64>::zeros([0, 2]).unwrap();
        let invalid_queries = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();
        let expected = vec![
            vec![Neighbor { index: 0, distance: 1.0 }, Neighbor { index: 1, distance: 1.0 }],
            vec![Neighbor { index: 1, distance: 0.25 }, Neighbor { index: 0, distance: 2.25 }],
        ];

        for algorithm in [
            KnnSearchAlgorithm::BruteForce,
            KnnSearchAlgorithm::KdTree,
            KnnSearchAlgorithm::BallTree,
        ] {
            let error_op = match algorithm {
                KnnSearchAlgorithm::BruteForce => "brute_force_knn_search",
                KnnSearchAlgorithm::KdTree => "kd_tree_search",
                KnnSearchAlgorithm::BallTree => "ball_tree_search",
                KnnSearchAlgorithm::Auto => unreachable!(),
            };
            let index = TrainingIndex::new(
                NDArray::from_shape_vec([3, 2], vec![-1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0]).unwrap(),
                algorithm,
                1,
            )
            .unwrap();

            assert_eq!(index.search_batch(&queries, 2), Ok(expected.clone()));
            assert_eq!(index.search_batch(&empty_queries, 2), Ok(Vec::new()));
            assert_eq!(
                index.search_batch(&invalid_queries, 2),
                Err(crate::AtlasMlError::ShapeMismatch {
                    op: error_op,
                    left: vec![2],
                    right: vec![1],
                    reason: "feature dimensions must match",
                })
            );
        }
    }
}
