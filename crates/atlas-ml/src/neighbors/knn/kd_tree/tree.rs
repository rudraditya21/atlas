use atlas_ndarray::OperandMetadata;

use super::node::KdTreeNode;
use crate::{
    AtlasMlError, AtlasMlResult,
    internal::row::LogicalRow,
    neighbors::knn::{
        metric::{axis_squared_distance_lower_bound, squared_distance_to_row},
        neighbor::Neighbor,
        neighbor_set::BoundedNeighborSet,
    },
};

const BUILD_OP: &str = "kd_tree_build";
const SEARCH_OP: &str = "kd_tree_search";

pub(crate) struct KdTree {
    root: KdTreeNode,
    sample_count: usize,
    feature_count: usize,
}

impl KdTree {
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

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let indices = (0..sample_count).collect();
        let root = if feature_count == 0 || sample_count <= leaf_size {
            KdTreeNode::leaf(indices)
        } else {
            build_node(features, indices, 0, leaf_size)
        };

        Ok(Self { root, sample_count, feature_count })
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

    pub(crate) fn root(&self) -> &KdTreeNode {
        &self.root
    }

    pub(crate) const fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(crate) const fn feature_count(&self) -> usize {
        self.feature_count
    }
}

fn build_node<F>(
    features: &F,
    mut indices: Vec<usize>,
    depth: usize,
    leaf_size: usize,
) -> KdTreeNode
where
    F: OperandMetadata<f64> + ?Sized,
{
    if indices.len() <= leaf_size {
        return KdTreeNode::leaf(indices);
    }

    let feature_count = features.shape()[1];
    let split_axis = depth % feature_count;
    indices.sort_unstable_by(|left, right| {
        feature(features, *left, split_axis)
            .total_cmp(&feature(features, *right, split_axis))
            .then_with(|| left.cmp(right))
    });

    let split_at = indices.len() / 2;
    let right_indices = indices.split_off(split_at);
    let pivot_index = right_indices[0];
    let left = build_node(features, indices, depth + 1, leaf_size);
    let right = build_node(features, right_indices, depth + 1, leaf_size);

    KdTreeNode::internal(split_axis, pivot_index, left, right)
}

pub(super) fn feature<F>(features: &F, row: usize, axis: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    features.data()[features.offset() + row * features.strides()[0] + axis * features.strides()[1]]
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

fn search_node<F>(
    node: &KdTreeNode,
    features: &F,
    query: &[f64],
    candidates: &mut BoundedNeighborSet,
) where
    F: OperandMetadata<f64> + ?Sized,
{
    if let Some(indices) = node.leaf_indices() {
        for &index in indices {
            candidates.insert(Neighbor {
                index,
                distance: squared_distance_to_row(LogicalRow::from_operand(features, index), query),
            });
        }
        return;
    }

    let split_axis = node.split_axis().expect("internal KD-tree nodes have a split axis");
    let pivot_index = node.pivot_index().expect("internal KD-tree nodes have a pivot index");
    let split_value = feature(features, pivot_index, split_axis);
    let (left, right) = node.children().expect("internal KD-tree nodes have two children");
    let (near, far) = if query[split_axis] <= split_value { (left, right) } else { (right, left) };

    search_node(near, features, query, candidates);

    let should_visit_far = !candidates.is_full()
        || axis_squared_distance_lower_bound(query[split_axis] - split_value)
            <= candidates.neighbors().last().unwrap().distance;
    if should_visit_far {
        search_node(far, features, query, candidates);
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{KdTree, KdTreeNode};
    use crate::neighbors::knn::brute_force::brute_force_search;

    #[test]
    fn builds_a_single_point_as_a_leaf() {
        let features = NDArray::from_shape_vec([1, 2], vec![1.0_f64, -1.0]).unwrap();
        let tree = KdTree::build(&features).unwrap();

        assert_eq!(tree.root().leaf_indices(), Some(&[0][..]));
    }

    #[test]
    fn splits_duplicate_points_by_training_index() {
        let features = NDArray::from_shape_vec([4, 2], vec![0.0_f64; 8]).unwrap();
        let tree = KdTree::build(&features).unwrap();

        assert_eq!(tree.root().split_axis(), Some(0));
        assert_eq!(tree.root().pivot_index(), Some(2));
        let (left, right) = tree.root().children().unwrap();
        assert_eq!(left.split_axis(), Some(1));
        assert_eq!(left.pivot_index(), Some(1));
        assert_eq!(right.split_axis(), Some(1));
        assert_eq!(right.pivot_index(), Some(3));
    }

    #[test]
    fn uses_each_axis_for_rectangular_feature_matrices() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![2.0_f64, 9.0, 0.0, 8.0, 1.0, 7.0]).unwrap();
        let tree = KdTree::build(&features).unwrap();

        assert_eq!(tree.root().split_axis(), Some(0));
        assert_eq!(tree.root().pivot_index(), Some(2));
        let (left, right) = tree.root().children().unwrap();
        assert_eq!(left.leaf_indices(), Some(&[1][..]));
        assert_eq!(right.split_axis(), Some(1));
        assert_eq!(right.pivot_index(), Some(0));
        let (right_left, right_right) = right.children().unwrap();
        assert_eq!(right_left.leaf_indices(), Some(&[2][..]));
        assert_eq!(right_right.leaf_indices(), Some(&[0][..]));
    }

    #[test]
    fn builds_stable_median_splits() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![2.0_f64, 0.0, 0.0, 2.0, 1.0, 3.0, 3.0, 1.0])
                .unwrap();
        let first = KdTree::build(&features).unwrap();
        let second = KdTree::build(&features).unwrap();

        assert_eq!(structure(first.root()), structure(second.root()));
    }

    #[test]
    fn matches_brute_force_neighbors_distances_ties_and_all_neighbor_counts() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![-1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0, 5.0, 5.0])
                .unwrap();
        let tree = KdTree::build(&features).unwrap();

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
        let tree = KdTree::build(&features).unwrap();
        let query = [1.5_f64, 1.5];

        for k in 1..=features.shape()[0] {
            assert_eq!(tree.search(&features, &query, k), brute_force_search(&features, &query, k));
        }
    }

    fn structure(node: &KdTreeNode) -> String {
        if let Some(indices) = node.leaf_indices() {
            return format!("{indices:?}");
        }

        let (left, right) = node.children().unwrap();
        format!(
            "{}:{}({},{})",
            node.split_axis().unwrap(),
            node.pivot_index().unwrap(),
            structure(left),
            structure(right)
        )
    }
}
