use std::sync::Arc;

use atlas_ndarray::{NDArray, OperandMetadata};

use super::{
    backend::{NeighborSearchBackend, build_search_backend_with_leaf_size},
    config::KnnSearchAlgorithm,
    metric::DistanceMetric,
    neighbor::Neighbor,
};
use crate::AtlasMlResult;

const QUERY_BLOCK_SIZE: usize = 32;

pub(crate) struct TrainingIndex {
    features: Arc<NDArray<f64>>,
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

        Ok(Self { features, backend })
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

    pub(crate) fn search_batch<Q>(
        &self,
        queries: &Q,
        k: usize,
        metric: &dyn DistanceMetric,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        let query_count = queries.shape()[0];
        let feature_count = queries.shape()[1];
        let mut batches = Vec::with_capacity(query_count);

        for block_start in (0..query_count).step_by(QUERY_BLOCK_SIZE) {
            let block_end = (block_start + QUERY_BLOCK_SIZE).min(query_count);
            if queries.strides()[1] == 1 {
                let query_rows = (block_start..block_end)
                    .map(|query_index| {
                        let start = queries.offset() + query_index * queries.strides()[0];
                        &queries.data()[start..start + feature_count]
                    })
                    .collect::<Vec<_>>();
                batches.extend(self.backend.search_batch(&query_rows, k, metric)?);
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
                batches.extend(self.backend.search_batch(&query_rows, k, metric)?);
            }
        }

        Ok(batches)
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
