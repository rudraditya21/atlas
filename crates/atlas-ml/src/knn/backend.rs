use std::sync::Arc;

use atlas_ndarray::NDArray;

use super::{
    config::KnnSearchAlgorithm, metric::DistanceMetric, neighbor::Neighbor,
    search::brute_force_search,
};
use crate::{AtlasMlError, AtlasMlResult};

pub(crate) trait NeighborSearchBackend: Send + Sync {
    fn algorithm(&self) -> KnnSearchAlgorithm;

    fn search(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Neighbor>>;
}

pub(crate) fn build_search_backend(
    features: Arc<NDArray<f64>>,
    algorithm: KnnSearchAlgorithm,
) -> AtlasMlResult<Box<dyn NeighborSearchBackend>> {
    match algorithm {
        KnnSearchAlgorithm::BruteForce | KnnSearchAlgorithm::Auto => {
            Ok(Box::new(BruteForceBackend { features }))
        }
        KnnSearchAlgorithm::KdTree | KnnSearchAlgorithm::BallTree => {
            Err(AtlasMlError::InvalidArgument {
                op: "knn_config",
                reason: "the requested search algorithm is not available",
            })
        }
    }
}

struct BruteForceBackend {
    features: Arc<NDArray<f64>>,
}

impl NeighborSearchBackend for BruteForceBackend {
    fn algorithm(&self) -> KnnSearchAlgorithm {
        KnnSearchAlgorithm::BruteForce
    }

    fn search(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Neighbor>> {
        brute_force_search(self.features.as_ref(), query, k, metric)
    }
}

#[cfg(test)]
pub(crate) fn assert_backend_equivalence(
    backend: &dyn NeighborSearchBackend,
    features: &NDArray<f64>,
    queries: &[&[f64]],
) {
    use super::{metric::SquaredEuclideanDistance, search::brute_force_search};

    let metric = SquaredEuclideanDistance;
    for query in queries {
        for k in 1..=features.shape()[0] {
            assert_eq!(
                backend.search(query, k, &metric),
                brute_force_search(features, query, k, &metric),
                "backend diverged for query {query:?} and k = {k}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use atlas_ndarray::NDArray;

    use super::{NeighborSearchBackend, assert_backend_equivalence, build_search_backend};
    use crate::knn::config::KnnSearchAlgorithm;

    #[test]
    fn brute_force_backend_satisfies_the_equivalence_contract() {
        let features = Arc::new(
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0])
                .unwrap(),
        );
        let backend =
            build_search_backend(Arc::clone(&features), KnnSearchAlgorithm::BruteForce).unwrap();
        let queries = [&[0.0_f64, 0.0][..], &[1.0_f64, 1.0][..]];

        assert_eq!(backend.algorithm(), KnnSearchAlgorithm::BruteForce);
        assert_backend_equivalence(backend.as_ref(), features.as_ref(), &queries);
    }

    #[test]
    fn rejects_backends_that_are_not_implemented() {
        let features = Arc::new(NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap());

        assert!(build_search_backend(features, KnnSearchAlgorithm::KdTree).is_err());
    }

    #[test]
    fn automatic_selection_resolves_to_brute_force() {
        let features = Arc::new(NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap());
        let backend = build_search_backend(features, KnnSearchAlgorithm::Auto).unwrap();

        assert_eq!(backend.algorithm(), KnnSearchAlgorithm::BruteForce);
    }
}
