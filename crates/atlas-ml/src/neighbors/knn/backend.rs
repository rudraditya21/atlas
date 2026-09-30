use atlas_ndarray::NDArray;

use super::{
    ball_tree::BallTree,
    brute_force::BruteForceSearch,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnSearchAlgorithm},
    kd_tree::KdTree,
};
use crate::AtlasMlResult;

pub(crate) enum NeighborSearchBackend {
    BruteForce(BruteForceSearch),
    KdTree(KdTree),
    BallTree(BallTree),
}

impl NeighborSearchBackend {
    pub(crate) fn new(
        features: &NDArray<f64>,
        algorithm: KnnSearchAlgorithm,
        tree_leaf_size: usize,
    ) -> AtlasMlResult<Self> {
        match algorithm {
            KnnSearchAlgorithm::BruteForce => Ok(Self::BruteForce(BruteForceSearch::new(features))),
            KnnSearchAlgorithm::KdTree => {
                Ok(Self::KdTree(KdTree::build_with_leaf_size(features, tree_leaf_size)?))
            }
            KnnSearchAlgorithm::BallTree => {
                Ok(Self::BallTree(BallTree::build_with_leaf_size(features, tree_leaf_size)?))
            }
            KnnSearchAlgorithm::Auto if features.shape()[0] <= AUTO_BRUTE_FORCE_MAX_SAMPLES => {
                Ok(Self::BruteForce(BruteForceSearch::new(features)))
            }
            KnnSearchAlgorithm::Auto => {
                Ok(Self::KdTree(KdTree::build_with_leaf_size(features, tree_leaf_size)?))
            }
        }
    }

    pub(crate) const fn algorithm(&self) -> KnnSearchAlgorithm {
        match self {
            Self::BruteForce(_) => KnnSearchAlgorithm::BruteForce,
            Self::KdTree(_) => KnnSearchAlgorithm::KdTree,
            Self::BallTree(_) => KnnSearchAlgorithm::BallTree,
        }
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::NeighborSearchBackend;
    use crate::neighbors::knn::{
        brute_force::brute_force_search,
        config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnSearchAlgorithm},
    };

    fn assert_backend_equivalence(
        backend: &NeighborSearchBackend,
        features: &NDArray<f64>,
        queries: &[&[f64]],
    ) {
        for query in queries {
            for k in 1..=features.shape()[0] {
                let actual = match backend {
                    NeighborSearchBackend::BruteForce(search) => search.search(features, query, k),
                    NeighborSearchBackend::KdTree(tree) => tree.search(features, query, k),
                    NeighborSearchBackend::BallTree(tree) => tree.search(features, query, k),
                };
                assert_eq!(
                    actual,
                    brute_force_search(features, query, k),
                    "backend diverged for query {query:?} and k = {k}"
                );
            }
        }
    }

    fn assert_all_backends_equivalent(features: &NDArray<f64>, queries: &[&[f64]]) {
        for algorithm in [
            KnnSearchAlgorithm::BruteForce,
            KnnSearchAlgorithm::KdTree,
            KnnSearchAlgorithm::BallTree,
        ] {
            let backend = NeighborSearchBackend::new(features, algorithm, 1).unwrap();
            assert_backend_equivalence(&backend, features, queries);
        }
    }

    fn seeded_values(mut state: u64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((state >> 11) as f64 / (1_u64 << 53) as f64) * 20.0 - 10.0
            })
            .collect()
    }

    #[test]
    fn explicit_backends_satisfy_the_equivalence_contract() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0])
                .unwrap();
        let queries = [&[0.0_f64, 0.0][..], &[1.0_f64, 1.0][..]];

        assert_all_backends_equivalent(&features, &queries);
    }

    #[test]
    fn automatic_selection_preserves_its_sample_threshold() {
        let small = NDArray::from_shape_vec(
            [AUTO_BRUTE_FORCE_MAX_SAMPLES, 1],
            vec![0.0_f64; AUTO_BRUTE_FORCE_MAX_SAMPLES],
        )
        .unwrap();
        let large_count = AUTO_BRUTE_FORCE_MAX_SAMPLES + 1;
        let large = NDArray::from_shape_vec(
            [large_count, 1],
            (0..large_count).map(|value| value as f64).collect(),
        )
        .unwrap();

        assert_eq!(
            NeighborSearchBackend::new(&small, KnnSearchAlgorithm::Auto, 1).unwrap().algorithm(),
            KnnSearchAlgorithm::BruteForce
        );
        let backend = NeighborSearchBackend::new(&large, KnnSearchAlgorithm::Auto, 1).unwrap();
        assert_eq!(backend.algorithm(), KnnSearchAlgorithm::KdTree);
        assert_backend_equivalence(&backend, &large, &[&[32.5_f64]]);
    }

    #[test]
    fn randomized_backends_match_brute_force() {
        const FEATURE_COUNT: usize = 3;
        const SAMPLE_COUNT: usize = 17;

        for seed in [1_u64, 7, 42, 1_337] {
            let features = NDArray::from_shape_vec(
                [SAMPLE_COUNT, FEATURE_COUNT],
                seeded_values(seed, SAMPLE_COUNT * FEATURE_COUNT),
            )
            .unwrap();
            let query_values = seeded_values(seed.wrapping_add(1), 4 * FEATURE_COUNT);
            let queries = query_values.chunks_exact(FEATURE_COUNT).collect::<Vec<_>>();

            assert_all_backends_equivalent(&features, &queries);
        }
    }

    #[test]
    fn backends_handle_single_samples_duplicates_and_leaf_sizes() {
        let single_sample = NDArray::from_shape_vec([1, 1], vec![3.0_f64]).unwrap();
        assert_all_backends_equivalent(&single_sample, &[&[3.0_f64]]);

        let features = NDArray::from_shape_vec(
            [5, 2],
            vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        )
        .unwrap();
        let queries = [&[0.0_f64, 0.0][..], &[0.5_f64, 0.0][..], &[2.0_f64, 0.0][..]];
        for leaf_size in 1..=features.shape()[0] + 1 {
            for algorithm in [KnnSearchAlgorithm::KdTree, KnnSearchAlgorithm::BallTree] {
                let backend = NeighborSearchBackend::new(&features, algorithm, leaf_size).unwrap();
                assert_backend_equivalence(&backend, &features, &queries);
            }
        }
    }
}
