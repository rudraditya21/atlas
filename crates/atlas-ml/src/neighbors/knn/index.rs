use atlas_ndarray::{NDArray, OperandMetadata};
use rayon::prelude::*;

use super::{backend::NeighborSearchBackend, config::KnnSearchAlgorithm, neighbor::Neighbor};
use crate::{AtlasMlResult, internal::parallel::should_parallelize_inference};

const QUERY_BLOCK_SIZE: usize = 32;

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
        match &self.backend {
            NeighborSearchBackend::BruteForce(search) => search.search(&self.features, query, k),
            NeighborSearchBackend::KdTree(tree) => tree.search(&self.features, query, k),
            NeighborSearchBackend::BallTree(tree) => tree.search(&self.features, query, k),
        }
    }

    pub(crate) fn search_batch<Q>(&self, queries: &Q, k: usize) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + Sync + ?Sized,
    {
        if let NeighborSearchBackend::BruteForce(search) = &self.backend {
            return search.search_batch(&self.features, queries, k);
        }

        let query_count = queries.shape()[0];
        let feature_count = queries.shape()[1];
        let block_starts = (0..query_count).step_by(QUERY_BLOCK_SIZE).collect::<Vec<_>>();
        let work_items =
            query_count.saturating_mul(self.features.shape()[0]).saturating_mul(feature_count);

        let blocks = if should_parallelize_inference(block_starts.len(), work_items) {
            block_starts
                .into_par_iter()
                .map(|block_start| self.search_block(queries, block_start, k))
                .collect::<AtlasMlResult<Vec<_>>>()?
        } else {
            block_starts
                .into_iter()
                .map(|block_start| self.search_block(queries, block_start, k))
                .collect::<AtlasMlResult<Vec<_>>>()?
        };

        Ok(blocks.into_iter().flatten().collect())
    }

    fn search_block<Q>(
        &self,
        queries: &Q,
        block_start: usize,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        let feature_count = queries.shape()[1];
        let block_end = (block_start + QUERY_BLOCK_SIZE).min(queries.shape()[0]);
        if queries.strides()[1] == 1 {
            let query_rows = (block_start..block_end)
                .map(|query_index| {
                    let start = queries.offset() + query_index * queries.strides()[0];
                    &queries.data()[start..start + feature_count]
                })
                .collect::<Vec<_>>();
            query_rows.iter().map(|query| self.search(query, k)).collect()
        } else {
            let block_len = block_end - block_start;
            let mut values = Vec::with_capacity(block_len * feature_count);
            for query_index in block_start..block_end {
                let row_offset = queries.offset() + query_index * queries.strides()[0];
                values.extend((0..feature_count).map(|feature_index| {
                    queries.data()[row_offset + feature_index * queries.strides()[1]]
                }));
            }
            let query_rows = (0..block_len)
                .map(|query_index| {
                    let start = query_index * feature_count;
                    &values[start..start + feature_count]
                })
                .collect::<Vec<_>>();
            query_rows.iter().map(|query| self.search(query, k)).collect()
        }
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
