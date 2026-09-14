use atlas_ndarray::OperandMetadata;

use super::{
    node::KdTreeNode,
    tree::{KdTree, feature},
};
use crate::{
    AtlasMlError, AtlasMlResult,
    core::row::copy_logical_row,
    knn::{metric::DistanceMetric, neighbor::Neighbor, neighbor_set::BoundedNeighborSet},
};

const SEARCH_OP: &str = "kd_tree_search";

impl KdTree {
    pub(crate) fn search<F, M>(
        &self,
        features: &F,
        query: &[f64],
        k: usize,
        metric: &M,
    ) -> AtlasMlResult<Vec<Neighbor>>
    where
        F: OperandMetadata<f64> + ?Sized,
        M: DistanceMetric + ?Sized,
    {
        validate_search_inputs(self, features, query, k)?;

        let mut candidates = BoundedNeighborSet::new(k);
        let mut row = vec![0.0; self.feature_count()];
        search_node(self.root(), features, query, metric, &mut row, &mut candidates);

        Ok(candidates.neighbors().to_vec())
    }
}

fn validate_search_inputs<F>(
    tree: &KdTree,
    features: &F,
    query: &[f64],
    k: usize,
) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
{
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op: SEARCH_OP,
            expected: "a rank-2 [samples, features] matrix",
            rank: features.ndim(),
        });
    }
    if features.shape()[0] != tree.sample_count() || features.shape()[1] != tree.feature_count() {
        return Err(AtlasMlError::ShapeMismatch {
            op: SEARCH_OP,
            left: features.shape().to_vec(),
            right: vec![tree.sample_count(), tree.feature_count()],
            reason: "training feature shape must match the KD-tree",
        });
    }
    if k == 0 || k > tree.sample_count() {
        return Err(AtlasMlError::InvalidArgument {
            op: SEARCH_OP,
            reason: "k must be between 1 and the number of training samples",
        });
    }
    if query.len() != tree.feature_count() {
        return Err(AtlasMlError::ShapeMismatch {
            op: SEARCH_OP,
            left: vec![tree.feature_count()],
            right: vec![query.len()],
            reason: "feature dimensions must match",
        });
    }

    Ok(())
}

fn search_node<F, M>(
    node: &KdTreeNode,
    features: &F,
    query: &[f64],
    metric: &M,
    row: &mut [f64],
    candidates: &mut BoundedNeighborSet,
) where
    F: OperandMetadata<f64> + ?Sized,
    M: DistanceMetric + ?Sized,
{
    if let Some(indices) = node.leaf_indices() {
        for &index in indices {
            copy_logical_row(features, index, row);
            candidates
                .insert(Neighbor { index, distance: metric.distance_same_dimension(row, query) });
        }
        return;
    }

    let split_axis = node.split_axis().expect("internal KD-tree nodes have a split axis");
    let pivot_index = node.pivot_index().expect("internal KD-tree nodes have a pivot index");
    let split_value = feature(features, pivot_index, split_axis);
    let (left, right) = node.children().expect("internal KD-tree nodes have two children");
    let (near, far) = if query[split_axis] <= split_value { (left, right) } else { (right, left) };

    search_node(near, features, query, metric, row, candidates);

    let should_visit_far = !candidates.is_full()
        || metric
            .axis_distance_lower_bound(query[split_axis] - split_value)
            .map_or(true, |lower_bound| {
                lower_bound <= candidates.neighbors().last().unwrap().distance
            });
    if should_visit_far {
        search_node(far, features, query, metric, row, candidates);
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::KdTree;
    use crate::knn::{metric::SquaredEuclideanDistance, search::brute_force_search};

    #[test]
    fn matches_brute_force_neighbors_distances_ties_and_all_neighbor_counts() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![-1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0, 5.0, 5.0])
                .unwrap();
        let tree = KdTree::build(&features).unwrap();

        for query in [&[0.0_f64, 0.0][..], &[4.0_f64, 4.0][..]] {
            for k in 1..=features.shape()[0] {
                assert_eq!(
                    tree.search(&features, query, k, &SquaredEuclideanDistance),
                    brute_force_search(&features, query, k, &SquaredEuclideanDistance)
                );
            }
        }
    }

    #[test]
    fn matches_brute_force_for_transposed_training_views() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 2.0, 8.0, 0.0, 2.0, 8.0]).unwrap();
        let features = values.view().transpose();
        let tree = KdTree::build(&features).unwrap();
        let query = [1.5_f64, 1.5];

        for k in 1..=features.shape()[0] {
            assert_eq!(
                tree.search(&features, &query, k, &SquaredEuclideanDistance),
                brute_force_search(&features, &query, k, &SquaredEuclideanDistance)
            );
        }
    }
}
