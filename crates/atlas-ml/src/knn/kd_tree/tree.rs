use atlas_ndarray::NDArray;

use super::node::KdTreeNode;
use crate::{AtlasMlError, AtlasMlResult};

const BUILD_OP: &str = "kd_tree_build";

pub(crate) struct KdTree {
    root: KdTreeNode,
}

impl KdTree {
    pub(crate) fn build(features: &NDArray<f64>) -> AtlasMlResult<Self> {
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

        let indices = (0..features.shape()[0]).collect();
        let root = if features.shape()[1] == 0 {
            KdTreeNode::leaf(indices)
        } else {
            build_node(features, indices, 0)
        };

        Ok(Self { root })
    }

    pub(crate) fn root(&self) -> &KdTreeNode {
        &self.root
    }
}

fn build_node(features: &NDArray<f64>, mut indices: Vec<usize>, depth: usize) -> KdTreeNode {
    if indices.len() == 1 {
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
    let left = build_node(features, indices, depth + 1);
    let right = build_node(features, right_indices, depth + 1);

    KdTreeNode::internal(split_axis, pivot_index, left, right)
}

fn feature(features: &NDArray<f64>, row: usize, axis: usize) -> f64 {
    features.data()[row * features.shape()[1] + axis]
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{KdTree, KdTreeNode};

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
