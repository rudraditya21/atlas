use atlas_ndarray::OperandMetadata;

use super::node::BallTreeNode;
use crate::{
    AtlasMlError, AtlasMlResult,
    internal::LogicalRow,
    neighbors::knn::{
        metric::{ball_squared_distance_lower_bound, squared_distance_to_row},
        top_k::{BoundedNeighborSet, Neighbor},
    },
};

const BUILD_OP: &str = "ball_tree_build";
const SEARCH_OP: &str = "ball_tree_search";

pub(crate) struct BallTree {
    root: BallTreeNode,
    sample_count: usize,
    feature_count: usize,
}

impl BallTree {
    #[cfg(test)]
    pub(crate) fn build<F>(features: &F) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        Self::build_with_leaf_size(features, 1)
    }

    pub(crate) fn build_with_leaf_size<F>(features: &F, leaf_size: usize) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        if leaf_size == 0 {
            return Err(AtlasMlError::InvalidArgument {
                op: BUILD_OP,
                reason: "leaf size must be positive",
            });
        }
        if features.ndim() != 2 {
            return Err(AtlasMlError::InvalidInputRank {
                op: BUILD_OP,
                expected: "a rank-2 [samples, features] matrix",
                rank: features.ndim(),
            });
        }
        if features.shape()[0] == 0 {
            return Err(AtlasMlError::EmptyInput { op: BUILD_OP });
        }

        Ok(Self {
            root: build_node(features, (0..features.shape()[0]).collect(), leaf_size),
            sample_count: features.shape()[0],
            feature_count: features.shape()[1],
        })
    }

    pub(crate) fn search<F>(
        &self,
        features: &F,
        query: &[f64],
        k: usize,
    ) -> AtlasMlResult<Vec<Neighbor>>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_search_inputs(self, features, query, k)?;

        let mut candidates = BoundedNeighborSet::new(k);
        search_node(self.root(), features, query, &mut candidates);

        Ok(candidates.neighbors().to_vec())
    }

    pub(crate) fn root(&self) -> &BallTreeNode {
        &self.root
    }

    pub(crate) const fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(crate) const fn feature_count(&self) -> usize {
        self.feature_count
    }
}

fn build_node<F>(features: &F, mut indices: Vec<usize>, leaf_size: usize) -> BallTreeNode
where
    F: OperandMetadata<f64> + ?Sized,
{
    indices.sort_unstable();
    let center = center(features, &indices);
    let radius = radius(features, &indices, &center);
    if indices.len() <= leaf_size {
        return BallTreeNode::leaf(center, radius, indices);
    }

    let (left_indices, right_indices) = farthest_point_partition(features, indices);
    let left = build_node(features, left_indices, leaf_size);
    let right = build_node(features, right_indices, leaf_size);

    BallTreeNode::internal(center, radius, left, right)
}

fn farthest_point_partition<F>(features: &F, indices: Vec<usize>) -> (Vec<usize>, Vec<usize>)
where
    F: OperandMetadata<f64> + ?Sized,
{
    let left_seed = indices[0];
    let right_seed = farthest_index(features, &indices, left_seed);
    let mut left = vec![left_seed];
    let mut right = vec![right_seed];

    for index in indices {
        if index == left_seed || index == right_seed {
            continue;
        }

        let left_distance = squared_point_distance(features, index, left_seed);
        let right_distance = squared_point_distance(features, index, right_seed);
        if left_distance < right_distance
            || (left_distance == right_distance && left.len() <= right.len())
        {
            left.push(index);
        } else {
            right.push(index);
        }
    }

    (left, right)
}

fn farthest_index<F>(features: &F, indices: &[usize], origin: usize) -> usize
where
    F: OperandMetadata<f64> + ?Sized,
{
    indices
        .iter()
        .copied()
        .filter(|&index| index != origin)
        .max_by(|left, right| {
            squared_point_distance(features, *left, origin)
                .total_cmp(&squared_point_distance(features, *right, origin))
                .then_with(|| right.cmp(left))
        })
        .expect("farthest-point partition requires at least two indices")
}

fn center<F>(features: &F, indices: &[usize]) -> Vec<f64>
where
    F: OperandMetadata<f64> + ?Sized,
{
    let mut center = vec![0.0; features.shape()[1]];
    for &index in indices {
        for (axis, value) in center.iter_mut().enumerate() {
            *value += feature(features, index, axis);
        }
    }
    for value in &mut center {
        *value /= indices.len() as f64;
    }

    center
}

fn radius<F>(features: &F, indices: &[usize], center: &[f64]) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    indices
        .iter()
        .map(|&index| squared_distance_to_center(features, index, center).sqrt())
        .fold(0.0, f64::max)
}

fn squared_point_distance<F>(features: &F, left: usize, right: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    (0..features.shape()[1])
        .map(|axis| {
            let delta = feature(features, left, axis) - feature(features, right, axis);
            delta * delta
        })
        .sum()
}

fn squared_distance_to_center<F>(features: &F, index: usize, center: &[f64]) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    center
        .iter()
        .enumerate()
        .map(|(axis, center_value)| {
            let delta = feature(features, index, axis) - center_value;
            delta * delta
        })
        .sum()
}

fn feature<F>(features: &F, row: usize, axis: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    features.data()[features.offset() + row * features.strides()[0] + axis * features.strides()[1]]
}

fn validate_search_inputs<F>(
    tree: &BallTree,
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
            reason: "training feature shape must match the Ball-tree",
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

fn search_node<F>(
    node: &BallTreeNode,
    features: &F,
    query: &[f64],
    candidates: &mut BoundedNeighborSet,
) where
    F: OperandMetadata<f64> + ?Sized,
{
    if let Some(indices) = node.leaf_indices() {
        for &index in indices {
            candidates.insert(
                index,
                squared_distance_to_row(LogicalRow::from_operand(features, index), query),
            );
        }
        return;
    }

    let (left, right) = node.children().expect("internal Ball-tree nodes have two children");
    let left_bound = ball_squared_distance_lower_bound(query, left.center(), left.radius());
    let right_bound = ball_squared_distance_lower_bound(query, right.center(), right.radius());
    let (near, far, far_bound) = if right_bound < left_bound {
        (right, left, left_bound)
    } else {
        (left, right, right_bound)
    };

    search_node(near, features, query, candidates);

    let should_visit_far = !candidates.is_full()
        || can_match_candidate(far_bound, candidates.neighbors().last().unwrap().distance);
    if should_visit_far {
        search_node(far, features, query, candidates);
    }
}

fn can_match_candidate(lower_bound: f64, worst_distance: f64) -> bool {
    lower_bound <= worst_distance
        || lower_bound - worst_distance
            <= 4.0 * f64::EPSILON * lower_bound.abs().max(worst_distance.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{BallTree, BallTreeNode};
    use crate::neighbors::knn::brute_force::brute_force_search;

    #[test]
    fn partitions_duplicate_points_deterministically() {
        let features = NDArray::from_shape_vec([4, 2], vec![1.0_f64; 8]).unwrap();
        let tree = BallTree::build(&features).unwrap();

        assert_eq!(tree.root().center(), &[1.0, 1.0]);
        assert_eq!(tree.root().radius(), 0.0);
        assert_eq!(leaf_indices(tree.root()), vec![0, 1, 2, 3]);
    }

    #[test]
    fn builds_uneven_farthest_point_partitions() {
        let features =
            NDArray::from_shape_vec([5, 1], vec![0.0_f64, 1.0, 2.0, 10.0, 11.0]).unwrap();
        let tree = BallTree::build(&features).unwrap();

        let (left, right) = tree.root().children().unwrap();
        assert_eq!(leaf_indices(left).len(), 3);
        assert_eq!(leaf_indices(right).len(), 2);
    }

    #[test]
    fn computes_centers_and_radii_for_high_dimensional_points() {
        let features =
            NDArray::from_shape_vec([2, 4], vec![0.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
                .unwrap();
        let tree = BallTree::build(&features).unwrap();

        assert_eq!(tree.root().center(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(tree.root().radius(), 1.0);
    }

    #[test]
    fn builds_deterministic_tree_layouts() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0])
                .unwrap();
        let first = BallTree::build(&features).unwrap();
        let second = BallTree::build(&features).unwrap();

        assert_eq!(structure(first.root()), structure(second.root()));
    }

    #[test]
    fn matches_brute_force_neighbors_distances_ties_and_all_neighbor_counts() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![-1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0, 5.0, 5.0])
                .unwrap();
        let tree = BallTree::build(&features).unwrap();

        for query in [&[0.0_f64, 0.0][..], &[4.0_f64, 4.0][..]] {
            for k in 1..=features.shape()[0] {
                assert_eq!(
                    tree.search(&features, query, k),
                    brute_force_search(&features, query, k)
                );
            }
        }
    }

    #[test]
    fn matches_brute_force_for_transposed_training_views() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 2.0, 8.0, 0.0, 2.0, 8.0]).unwrap();
        let features = values.view().transpose();
        let tree = BallTree::build(&features).unwrap();
        let query = [1.5_f64, 1.5];

        for k in 1..=features.shape()[0] {
            assert_eq!(tree.search(&features, &query, k), brute_force_search(&features, &query, k));
        }
    }

    #[test]
    fn preserves_training_index_ties_at_ball_boundaries() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0]).unwrap();
        let tree = BallTree::build(&features).unwrap();
        let query = [1.0_f64, 1.0];

        assert_eq!(tree.search(&features, &query, 2), brute_force_search(&features, &query, 2));
    }

    fn leaf_indices(node: &BallTreeNode) -> Vec<usize> {
        if let Some(indices) = node.leaf_indices() {
            return indices.to_vec();
        }

        let (left, right) = node.children().unwrap();
        let mut indices = leaf_indices(left);
        indices.extend(leaf_indices(right));
        indices.sort_unstable();
        indices
    }

    fn structure(node: &BallTreeNode) -> String {
        if let Some(indices) = node.leaf_indices() {
            return format!("{:?}:{indices:?}", node.center());
        }

        let (left, right) = node.children().unwrap();
        format!("{:?}:{}({},{})", node.center(), node.radius(), structure(left), structure(right))
    }
}
