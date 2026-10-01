use std::mem::size_of;

use atlas_ndarray::{NDArray, OperandMetadata};

use super::{
    metric::squared_distance_to_row,
    top_k::{BoundedNeighborSet, Neighbor},
};
use crate::{
    AtlasMlError, AtlasMlResult,
    internal::{LogicalRow, copy_logical_row},
};

const QUERY_BLOCK_ROWS: usize = 64;
const TRAINING_BLOCK_TARGET_BYTES: usize = 4 * 1024 * 1024;
const OP: &str = "brute_force_knn_search";

pub(crate) struct BruteForceSearch {
    training_squared_norms: Box<[f64]>,
}

impl BruteForceSearch {
    pub(crate) fn new(training_features: &NDArray<f64>) -> Self {
        Self { training_squared_norms: squared_row_norms(training_features).into_boxed_slice() }
    }

    pub(crate) fn search(
        &self,
        training_features: &NDArray<f64>,
        query: &[f64],
        k: usize,
    ) -> AtlasMlResult<Vec<Neighbor>> {
        validate_search(training_features, query.len(), k)?;
        Ok(scalar_search(training_features, query, k))
    }

    pub(crate) fn search_batch<Q>(
        &self,
        training_features: &NDArray<f64>,
        queries: &Q,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + Sync + ?Sized,
    {
        if queries.ndim() != 2 {
            return Err(AtlasMlError::InvalidInputRank {
                op: OP,
                expected: "a rank-2 [samples, features] matrix",
                rank: queries.ndim(),
            });
        }
        validate_search(training_features, queries.shape()[1], k)?;

        let query_count = queries.shape()[0];
        if query_count == 0 {
            return Ok(Vec::new());
        }
        if query_count == 1 {
            let mut query = vec![0.0; queries.shape()[1]];
            copy_logical_row(queries, 0, &mut query);
            return Ok(vec![scalar_search(training_features, &query, k)]);
        }

        self.search_blocked(training_features, queries, k)
    }

    fn search_blocked<Q>(
        &self,
        training_features: &NDArray<f64>,
        queries: &Q,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + Sync + ?Sized,
    {
        let query_squared_norms = squared_row_norms(queries);
        let query_count = queries.shape()[0];
        let sample_count = training_features.shape()[0];
        let feature_count = training_features.shape()[1];
        let query_block_size = query_count.min(QUERY_BLOCK_ROWS);
        let training_block_size =
            distance_training_block_size(sample_count, feature_count, query_block_size);
        let block_starts = (0..query_count).step_by(query_block_size).collect::<Vec<_>>();
        let blocks = block_starts
            .into_iter()
            .map(|block_start| {
                self.search_block(
                    training_features,
                    queries,
                    &query_squared_norms,
                    block_start,
                    query_block_size,
                    training_block_size,
                    k,
                )
            })
            .collect::<AtlasMlResult<Vec<_>>>()?;

        Ok(blocks.into_iter().flatten().collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn search_block<Q>(
        &self,
        training_features: &NDArray<f64>,
        queries: &Q,
        query_squared_norms: &[f64],
        block_start: usize,
        query_block_size: usize,
        training_block_size: usize,
        k: usize,
    ) -> AtlasMlResult<Vec<Vec<Neighbor>>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        let feature_count = queries.shape()[1];
        let block_end = (block_start + query_block_size).min(queries.shape()[0]);
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
        let sample_count = training_features.shape()[0];
        let mut neighbor_sets =
            (0..block_len).map(|_| BoundedNeighborSet::new(k)).collect::<Vec<_>>();

        for training_start in (0..sample_count).step_by(training_block_size) {
            let training_len = training_block_size.min(sample_count - training_start);
            let training_block = training_features
                .view()
                .slice([training_start, 0], [training_len, feature_count])?;
            let products = atlas_linalg::matmul(&query_block, training_block.transpose())?;

            for (block_row, neighbors) in neighbor_sets.iter_mut().enumerate() {
                let query_norm = query_squared_norms[block_start + block_row];
                let product_row =
                    &products.data()[block_row * training_len..(block_row + 1) * training_len];
                let training_norms =
                    &self.training_squared_norms[training_start..training_start + training_len];
                neighbors.insert_distance_block(
                    training_start,
                    training_norms.iter().zip(product_row).map(|(&training_norm, &product)| {
                        let distance = query_norm + training_norm - 2.0 * product;
                        if distance < 0.0 { 0.0 } else { distance }
                    }),
                );
            }
        }

        Ok(neighbor_sets.into_iter().map(|neighbors| neighbors.neighbors().to_vec()).collect())
    }
}

pub(super) fn scalar_search<F>(training_features: &F, query: &[f64], k: usize) -> Vec<Neighbor>
where
    F: OperandMetadata<f64> + ?Sized,
{
    let mut neighbors = BoundedNeighborSet::new(k);
    neighbors.insert_distance_block(
        0,
        (0..training_features.shape()[0]).map(|sample_index| {
            squared_distance_to_row(
                LogicalRow::from_operand(training_features, sample_index),
                query,
            )
        }),
    );
    neighbors.neighbors().to_vec()
}

fn validate_search<F>(
    training_features: &F,
    query_feature_count: usize,
    k: usize,
) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
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
    if query_feature_count != feature_count {
        return Err(AtlasMlError::ShapeMismatch {
            op: OP,
            left: vec![feature_count],
            right: vec![query_feature_count],
            reason: "feature dimensions must match",
        });
    }

    Ok(())
}

fn distance_training_block_size(
    training_count: usize,
    feature_count: usize,
    query_count: usize,
) -> usize {
    let bytes_per_training_sample =
        feature_count.saturating_add(query_count).saturating_mul(size_of::<f64>());
    TRAINING_BLOCK_TARGET_BYTES
        .checked_div(bytes_per_training_sample)
        .unwrap_or(1)
        .max(1)
        .min(training_count.max(1))
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
    use atlas_ndarray::NDArray;

    use super::scalar_search;
    use crate::neighbors::knn::top_k::Neighbor;

    fn neighbor(index: usize, distance: f64) -> Neighbor {
        Neighbor { index, distance }
    }

    #[test]
    fn finds_exact_nearest_neighbors() {
        let training =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0])
                .unwrap();

        assert_eq!(
            scalar_search(&training, &[1.5, 1.5], 2),
            vec![neighbor(1, 0.5), neighbor(0, 4.5)]
        );
    }

    #[test]
    fn searches_logical_rows_of_transposed_views() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 2.0, 8.0, 0.0, 2.0, 8.0]).unwrap();
        let training = values.view().transpose();

        assert_eq!(
            scalar_search(&training, &[1.5, 1.5], 2),
            vec![neighbor(1, 0.5), neighbor(0, 4.5)]
        );
    }

    #[test]
    fn resolves_equal_distances_by_training_index() {
        let training =
            NDArray::from_shape_vec([3, 2], vec![-1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0]).unwrap();

        assert_eq!(
            scalar_search(&training, &[0.0, 0.0], 2),
            vec![neighbor(0, 1.0), neighbor(1, 1.0)]
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
            assert_eq!(scalar_search(&training, &[0.0], k), neighbors);
        }
    }
}
