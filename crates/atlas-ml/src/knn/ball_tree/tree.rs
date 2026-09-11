use atlas_ndarray::OperandMetadata;

use super::node::BallTreeNode;
use crate::{AtlasMlError, AtlasMlResult};

const BUILD_OP: &str = "ball_tree_build";

pub(crate) struct BallTree {
    root: BallTreeNode,
    sample_count: usize,
    feature_count: usize,
}

impl BallTree {
    pub(crate) fn build<F>(features: &F) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
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
            root: build_node(features, (0..features.shape()[0]).collect()),
            sample_count: features.shape()[0],
            feature_count: features.shape()[1],
        })
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

fn build_node<F>(features: &F, mut indices: Vec<usize>) -> BallTreeNode
where
    F: OperandMetadata<f64> + ?Sized,
{
    indices.sort_unstable();
    let center = center(features, &indices);
    let radius = radius(features, &indices, &center);
    if indices.len() == 1 {
        return BallTreeNode::leaf(center, radius, indices);
    }

    let (left_indices, right_indices) = farthest_point_partition(features, indices);
    let left = build_node(features, left_indices);
    let right = build_node(features, right_indices);

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

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{BallTree, BallTreeNode};

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
