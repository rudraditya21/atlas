use std::sync::Arc;

use atlas_ndarray::NDArray;

use super::{
    backend::{NeighborSearchBackend, build_search_backend},
    config::KnnSearchAlgorithm,
    metric::DistanceMetric,
    neighbor::Neighbor,
};
use crate::AtlasMlResult;

pub(crate) struct TrainingIndex {
    features: Arc<NDArray<f64>>,
    backend: Box<dyn NeighborSearchBackend>,
}

impl TrainingIndex {
    pub(crate) fn new(features: NDArray<f64>, algorithm: KnnSearchAlgorithm) -> Self {
        let features = Arc::new(features);
        let backend = build_search_backend(Arc::clone(&features), algorithm);

        Self { features, backend }
    }

    pub(crate) fn features(&self) -> &Arc<NDArray<f64>> {
        &self.features
    }

    pub(crate) fn search_algorithm(&self) -> KnnSearchAlgorithm {
        self.backend.algorithm()
    }

    pub(crate) fn search(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Neighbor>> {
        self.backend.search(query, k, metric)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use atlas_ndarray::NDArray;

    use super::TrainingIndex;
    use crate::knn::{
        config::KnnSearchAlgorithm, metric::SquaredEuclideanDistance, neighbor::Neighbor,
    };

    #[test]
    fn retains_shared_owned_training_features_and_shape() {
        let index = TrainingIndex::new(
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            KnnSearchAlgorithm::BruteForce,
        );

        assert_eq!(index.features().shape(), &[2, 3]);
        assert_eq!(index.features().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(Arc::strong_count(index.features()), 2);
    }

    #[test]
    fn initializes_the_requested_search_backend() {
        let index = TrainingIndex::new(
            NDArray::from_shape_vec([2, 1], vec![0.0_f64, 2.0]).unwrap(),
            KnnSearchAlgorithm::BruteForce,
        );

        assert_eq!(index.search_algorithm(), KnnSearchAlgorithm::BruteForce);
        assert_eq!(
            index.search(&[1.0], 1, &SquaredEuclideanDistance),
            Ok(vec![Neighbor { index: 0, distance: 1.0 }])
        );
    }
}
