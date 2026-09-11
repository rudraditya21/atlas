#[derive(Debug)]
pub(crate) enum BallTreeNode {
    Leaf { center: Box<[f64]>, radius: f64, indices: Box<[usize]> },
    Internal { center: Box<[f64]>, radius: f64, left: Box<BallTreeNode>, right: Box<BallTreeNode> },
}

impl BallTreeNode {
    pub(crate) fn leaf(center: Vec<f64>, radius: f64, mut indices: Vec<usize>) -> Self {
        validate_radius(radius);
        indices.sort_unstable();
        assert!(!indices.is_empty(), "Ball-tree leaves must contain at least one training index");
        assert!(
            indices.windows(2).all(|pair| pair[0] != pair[1]),
            "Ball-tree leaf training indices must be unique"
        );

        Self::Leaf { center: center.into(), radius, indices: indices.into() }
    }

    pub(crate) fn internal(center: Vec<f64>, radius: f64, left: Self, right: Self) -> Self {
        validate_radius(radius);
        assert_eq!(
            left.center().len(),
            center.len(),
            "Ball-tree child centers must match the parent dimension"
        );
        assert_eq!(
            right.center().len(),
            center.len(),
            "Ball-tree child centers must match the parent dimension"
        );

        Self::Internal {
            center: center.into(),
            radius,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub(crate) fn center(&self) -> &[f64] {
        match self {
            Self::Leaf { center, .. } | Self::Internal { center, .. } => center,
        }
    }

    pub(crate) const fn radius(&self) -> f64 {
        match self {
            Self::Leaf { radius, .. } | Self::Internal { radius, .. } => *radius,
        }
    }

    pub(crate) const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    pub(crate) fn leaf_indices(&self) -> Option<&[usize]> {
        match self {
            Self::Leaf { indices, .. } => Some(indices),
            Self::Internal { .. } => None,
        }
    }

    pub(crate) fn children(&self) -> Option<(&Self, &Self)> {
        match self {
            Self::Leaf { .. } => None,
            Self::Internal { left, right, .. } => Some((left, right)),
        }
    }
}

fn validate_radius(radius: f64) {
    assert!(
        radius.is_finite() && radius >= 0.0,
        "Ball-tree node radii must be finite and nonnegative"
    );
}

#[cfg(test)]
mod tests {
    use super::BallTreeNode;

    #[test]
    fn leaves_preserve_center_radius_and_deterministic_indices() {
        let leaf = BallTreeNode::leaf(vec![1.0, -1.0], 2.5, vec![4, 1, 3]);

        assert!(leaf.is_leaf());
        assert_eq!(leaf.center(), &[1.0, -1.0]);
        assert_eq!(leaf.radius(), 2.5);
        assert_eq!(leaf.leaf_indices(), Some(&[1, 3, 4][..]));
        assert!(leaf.children().is_none());
    }

    #[test]
    fn internal_nodes_require_and_retain_both_children() {
        let node = BallTreeNode::internal(
            vec![0.0],
            2.0,
            BallTreeNode::leaf(vec![0.0], 0.0, vec![0]),
            BallTreeNode::leaf(vec![1.0], 0.0, vec![1]),
        );

        assert!(!node.is_leaf());
        let (left, right) = node.children().unwrap();
        assert_eq!(left.leaf_indices(), Some(&[0][..]));
        assert_eq!(right.leaf_indices(), Some(&[1][..]));
    }

    #[test]
    #[should_panic(expected = "finite and nonnegative")]
    fn nodes_reject_invalid_radius_bounds() {
        BallTreeNode::leaf(vec![0.0], -1.0, vec![0]);
    }

    #[test]
    #[should_panic(expected = "at least one training index")]
    fn leaves_reject_empty_index_storage() {
        BallTreeNode::leaf(vec![0.0], 0.0, Vec::new());
    }
}
