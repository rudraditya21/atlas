use atlas_ndarray::{NDArray, OperandMetadata};
use rayon::prelude::*;

use super::{
    ball_tree::BallTree,
    brute_force::BruteForceSearch,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnSearchAlgorithm},
    kd_tree::KdTree,
};
use crate::{AtlasMlResult, internal::parallel::should_parallelize_inference};

const QUERY_BLOCK_SIZE: usize = 32;

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

    pub(crate) const fn search_op(&self) -> &'static str {
        match self {
            Self::BruteForce(_) => "brute_force_knn_search",
            Self::KdTree(_) => "kd_tree_search",
            Self::BallTree(_) => "ball_tree_search",
        }
    }

    pub(crate) fn search(
        &self,
        features: &NDArray<f64>,
        query: &[f64],
        k: usize,
    ) -> AtlasMlResult<Vec<super::neighbor::Neighbor>> {
        match self {
            Self::BruteForce(search) => search.search(features, query, k),
            Self::KdTree(tree) => tree.search(features, query, k),
            Self::BallTree(tree) => tree.search(features, query, k),
        }
    }

    pub(crate) fn search_batch<Q>(
        &self,
        features: &NDArray<f64>,
        queries: &Q,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<super::neighbor::Neighbor>>>
    where
        Q: OperandMetadata<f64> + Sync + ?Sized,
    {
        if let Self::BruteForce(search) = self {
            return search.search_batch(features, queries, k);
        }

        let query_count = queries.shape()[0];
        let feature_count = queries.shape()[1];
        let block_starts = (0..query_count).step_by(QUERY_BLOCK_SIZE).collect::<Vec<_>>();
        let work_items =
            query_count.saturating_mul(features.shape()[0]).saturating_mul(feature_count);

        let blocks = if should_parallelize_inference(block_starts.len(), work_items) {
            block_starts
                .into_par_iter()
                .map(|block_start| self.search_block(features, queries, block_start, k))
                .collect::<AtlasMlResult<Vec<_>>>()?
        } else {
            block_starts
                .into_iter()
                .map(|block_start| self.search_block(features, queries, block_start, k))
                .collect::<AtlasMlResult<Vec<_>>>()?
        };

        Ok(blocks.into_iter().flatten().collect())
    }

    fn search_block<Q>(
        &self,
        features: &NDArray<f64>,
        queries: &Q,
        block_start: usize,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<super::neighbor::Neighbor>>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        let feature_count = queries.shape()[1];
        let block_end = (block_start + QUERY_BLOCK_SIZE).min(queries.shape()[0]);
        if queries.strides()[1] == 1 {
            (block_start..block_end)
                .map(|query_index| {
                    let start = queries.offset() + query_index * queries.strides()[0];
                    self.search(features, &queries.data()[start..start + feature_count], k)
                })
                .collect()
        } else {
            let mut query = vec![0.0; feature_count];
            (block_start..block_end)
                .map(|query_index| {
                    crate::internal::row::copy_logical_row(queries, query_index, &mut query);
                    self.search(features, &query, k)
                })
                .collect()
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
