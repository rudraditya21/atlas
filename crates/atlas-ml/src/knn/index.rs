use std::sync::Arc;

use atlas_ndarray::{NDArray, OperandMetadata};
use rayon::prelude::*;

use super::{
    backend::{NeighborSearchBackend, build_search_backend_with_leaf_size},
    config::KnnSearchAlgorithm,
    metric::DistanceMetric,
    neighbor::Neighbor,
    neighbor_set::BoundedNeighborSet,
};
use crate::{
    AtlasMlError, AtlasMlResult,
    core::{parallel::should_parallelize_inference, row::copy_logical_row},
};

const QUERY_BLOCK_SIZE: usize = 32;
const BRUTE_FORCE_OP: &str = "brute_force_knn_search";

pub(crate) struct TrainingIndex {
    features: Arc<NDArray<f64>>,
    training_squared_norms: Box<[f64]>,
    backend: Box<dyn NeighborSearchBackend>,
}

impl TrainingIndex {
    pub(crate) fn new(
        features: NDArray<f64>,
        algorithm: KnnSearchAlgorithm,
        tree_leaf_size: usize,
    ) -> AtlasMlResult<Self> {
        let features = Arc::new(features);
        let backend =
            build_search_backend_with_leaf_size(Arc::clone(&features), algorithm, tree_leaf_size)?;
        let training_squared_norms = if backend.algorithm() == KnnSearchAlgorithm::BruteForce {
            squared_row_norms(features.as_ref()).into_boxed_slice()
        } else {
            Box::default()
        };

        Ok(Self { features, training_squared_norms, backend })
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
        if self.backend.algorithm() == KnnSearchAlgorithm::BruteForce
            && metric.supports_squared_euclidean_expansion()
        {
            let queries = NDArray::from_shape_vec([1, query.len()], query.to_vec())?;
            return Ok(self
                .search_gemm_batch(&queries, k)?
                .pop()
                .expect("one query produces one neighbor batch"));
        }

        self.backend.search(query, k, metric)
    }

    pub(crate) fn search_batch<Q>(
        &self,
        queries: &Q,
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + Sync + ?Sized,
    {
        if self.backend.algorithm() == KnnSearchAlgorithm::BruteForce
            && metric.supports_squared_euclidean_expansion()
        {
            return self.search_gemm_batch(queries, k);
        }

        let query_count = queries.shape()[0];
        let feature_count = queries.shape()[1];
        let block_starts = (0..query_count).step_by(QUERY_BLOCK_SIZE).collect::<Vec<_>>();
        let work_items =
            query_count.saturating_mul(self.features.shape()[0]).saturating_mul(feature_count);

        let blocks = if should_parallelize_inference(block_starts.len(), work_items) {
            block_starts
                .into_par_iter()
                .map(|block_start| self.search_block(queries, block_start, k, metric))
                .collect::<AtlasMlResult<Vec<_>>>()?
        } else {
            block_starts
                .into_iter()
                .map(|block_start| self.search_block(queries, block_start, k, metric))
                .collect::<AtlasMlResult<Vec<_>>>()?
        };

        Ok(blocks.into_iter().flatten().collect())
    }

    fn search_gemm_batch<Q>(&self, queries: &Q, k: usize) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        if queries.ndim() != 2 {
            return Err(AtlasMlError::InvalidInputRank {
                op: BRUTE_FORCE_OP,
                expected: "a rank-2 [samples, features] matrix",
                rank: queries.ndim(),
            });
        }

        let sample_count = self.features.shape()[0];
        if k == 0 || k > sample_count {
            return Err(AtlasMlError::InvalidArgument {
                op: BRUTE_FORCE_OP,
                reason: "k must be between 1 and the number of training samples",
            });
        }

        let feature_count = self.features.shape()[1];
        if queries.shape()[1] != feature_count {
            return Err(AtlasMlError::ShapeMismatch {
                op: BRUTE_FORCE_OP,
                left: vec![feature_count],
                right: vec![queries.shape()[1]],
                reason: "feature dimensions must match",
            });
        }

        let query_squared_norms = squared_row_norms(queries);
        let mut neighbors = Vec::with_capacity(queries.shape()[0]);
        for block_start in (0..queries.shape()[0]).step_by(QUERY_BLOCK_SIZE) {
            neighbors.extend(self.search_gemm_block(
                queries,
                &query_squared_norms,
                block_start,
                k,
            )?);
        }
        Ok(neighbors)
    }

    fn search_gemm_block<Q>(
        &self,
        queries: &Q,
        query_squared_norms: &[f64],
        block_start: usize,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        let feature_count = queries.shape()[1];
        let block_end = (block_start + QUERY_BLOCK_SIZE).min(queries.shape()[0]);
        let block_len = block_end - block_start;
        let mut query_values = vec![0.0; block_len * feature_count];
        for (block_row, query_index) in (block_start..block_end).enumerate() {
            let row_start = block_row * feature_count;
            copy_logical_row(
                queries,
                query_index,
                &mut query_values[row_start..row_start + feature_count],
            );
        }

        let query_block = NDArray::from_shape_vec([block_len, feature_count], query_values)?;
        let products = atlas_linalg::matmul(&query_block, self.features.view().transpose())?;
        let sample_count = self.features.shape()[0];

        Ok((0..block_len)
            .map(|block_row| {
                let query_norm = query_squared_norms[block_start + block_row];
                let product_row =
                    &products.data()[block_row * sample_count..(block_row + 1) * sample_count];
                let mut neighbors = BoundedNeighborSet::new(k);
                for (sample_index, (&training_norm, &product)) in
                    self.training_squared_norms.iter().zip(product_row).enumerate()
                {
                    let distance = query_norm + training_norm - 2.0 * product;
                    neighbors.insert(Neighbor {
                        index: sample_index,
                        distance: if distance < 0.0 { 0.0 } else { distance },
                    });
                }
                neighbors.neighbors().to_vec()
            })
            .collect())
    }

    fn search_block<Q>(
        &self,
        queries: &Q,
        block_start: usize,
        k: usize,
        metric: &dyn DistanceMetric,
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
            self.backend.search_batch(&query_rows, k, metric)
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
            self.backend.search_batch(&query_rows, k, metric)
        }
    }
}

fn squared_row_norms<O>(values: &O) -> Vec<f64>
where
    O: OperandMetadata<f64> + ?Sized,
{
    let row_count = values.shape()[0];
    let column_count = values.shape()[1];
    (0..row_count)
        .map(|row| {
            let row_offset = values.offset() + row * values.strides()[0];
            (0..column_count)
                .map(|column| {
                    let value = values.data()[row_offset + column * values.strides()[1]];
                    value * value
                })
                .sum()
        })
        .collect()
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
            1,
        )
        .unwrap();

        assert_eq!(index.features().shape(), &[2, 3]);
        assert_eq!(index.features().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(Arc::strong_count(index.features()), 2);
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
        assert_eq!(
            index.search(&[1.0], 1, &SquaredEuclideanDistance),
            Ok(vec![Neighbor { index: 0, distance: 1.0 }])
        );
    }
}
