use std::sync::Arc;

use atlas_ndarray::NDArray;

use super::{
    ball_tree::tree::BallTree, config::KnnSearchAlgorithm, kd_tree::tree::KdTree,
    metric::DistanceMetric, neighbor::Neighbor, search::brute_force_search,
};
use crate::AtlasMlResult;

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
        KnnSearchAlgorithm::KdTree => Ok(Box::new(KdTreeBackend::new(features)?)),
        KnnSearchAlgorithm::BallTree => Ok(Box::new(BallTreeBackend::new(features)?)),
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

struct KdTreeBackend {
    features: Arc<NDArray<f64>>,
    tree: KdTree,
}

impl KdTreeBackend {
    fn new(features: Arc<NDArray<f64>>) -> AtlasMlResult<Self> {
        let tree = KdTree::build(features.as_ref())?;

        Ok(Self { features, tree })
    }
}

impl NeighborSearchBackend for KdTreeBackend {
    fn algorithm(&self) -> KnnSearchAlgorithm {
        KnnSearchAlgorithm::KdTree
    }

    fn search(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Neighbor>> {
        self.tree.search(self.features.as_ref(), query, k, metric)
    }
}

struct BallTreeBackend {
    features: Arc<NDArray<f64>>,
    tree: BallTree,
}

impl BallTreeBackend {
    fn new(features: Arc<NDArray<f64>>) -> AtlasMlResult<Self> {
        let tree = BallTree::build(features.as_ref())?;

        Ok(Self { features, tree })
    }
}

impl NeighborSearchBackend for BallTreeBackend {
    fn algorithm(&self) -> KnnSearchAlgorithm {
        KnnSearchAlgorithm::BallTree
    }

    fn search(
        &self,
        query: &[f64],
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Neighbor>> {
        self.tree.search(self.features.as_ref(), query, k, metric)
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
    fn kd_tree_backend_satisfies_the_equivalence_contract() {
        let features = Arc::new(
            NDArray::from_shape_vec([3, 2], vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0]).unwrap(),
        );
        let backend =
            build_search_backend(Arc::clone(&features), KnnSearchAlgorithm::KdTree).unwrap();
        let queries = [&[0.0_f64, 0.0][..], &[1.0_f64, 1.0][..]];

        assert_eq!(backend.algorithm(), KnnSearchAlgorithm::KdTree);
        assert_backend_equivalence(backend.as_ref(), features.as_ref(), &queries);
    }

    #[test]
    fn ball_tree_backend_satisfies_the_equivalence_contract() {
        let features = Arc::new(
            NDArray::from_shape_vec([3, 2], vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0]).unwrap(),
        );
        let backend =
            build_search_backend(Arc::clone(&features), KnnSearchAlgorithm::BallTree).unwrap();
        let queries = [&[0.0_f64, 0.0][..], &[1.0_f64, 1.0][..]];

        assert_eq!(backend.algorithm(), KnnSearchAlgorithm::BallTree);
        assert_backend_equivalence(backend.as_ref(), features.as_ref(), &queries);
    }

    #[test]
    fn automatic_selection_resolves_to_brute_force() {
        let features = Arc::new(NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap());
        let backend = build_search_backend(features, KnnSearchAlgorithm::Auto).unwrap();

        assert_eq!(backend.algorithm(), KnnSearchAlgorithm::BruteForce);
    }
}
