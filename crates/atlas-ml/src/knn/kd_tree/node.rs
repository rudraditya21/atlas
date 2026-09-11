#[derive(Debug)]
pub(crate) enum KdTreeNode {
    Leaf {
        indices: Box<[usize]>,
    },
    Internal {
        split_axis: usize,
        pivot_index: usize,
        left: Box<KdTreeNode>,
        right: Box<KdTreeNode>,
    },
}

impl KdTreeNode {
    pub(crate) fn leaf(mut indices: Vec<usize>) -> Self {
        indices.sort_unstable();
        assert!(!indices.is_empty(), "KD-tree leaves must contain at least one training index");
        assert!(
            indices.windows(2).all(|pair| pair[0] != pair[1]),
            "KD-tree leaf training indices must be unique"
        );

        Self::Leaf { indices: indices.into() }
    }

    pub(crate) fn internal(split_axis: usize, pivot_index: usize, left: Self, right: Self) -> Self {
        Self::Internal { split_axis, pivot_index, left: Box::new(left), right: Box::new(right) }
    }

    pub(crate) const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    pub(crate) const fn split_axis(&self) -> Option<usize> {
        match self {
            Self::Leaf { .. } => None,
            Self::Internal { split_axis, .. } => Some(*split_axis),
        }
    }

    pub(crate) const fn pivot_index(&self) -> Option<usize> {
        match self {
            Self::Leaf { .. } => None,
            Self::Internal { pivot_index, .. } => Some(*pivot_index),
        }
    }

    pub(crate) fn leaf_indices(&self) -> Option<&[usize]> {
        match self {
            Self::Leaf { indices } => Some(indices),
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

#[cfg(test)]
mod tests {
    use super::KdTreeNode;

    #[test]
    fn leaves_store_training_indices_in_deterministic_order() {
        let leaf = KdTreeNode::leaf(vec![7, 2, 5]);

        assert!(leaf.is_leaf());
        assert_eq!(leaf.leaf_indices(), Some(&[2, 5, 7][..]));
        assert_eq!(leaf.split_axis(), None);
        assert_eq!(leaf.pivot_index(), None);
        assert!(leaf.children().is_none());
    }

    #[test]
    fn internal_nodes_retain_split_metadata_and_children() {
        let node = KdTreeNode::internal(1, 4, KdTreeNode::leaf(vec![2]), KdTreeNode::leaf(vec![7]));

        assert!(!node.is_leaf());
        assert_eq!(node.split_axis(), Some(1));
        assert_eq!(node.pivot_index(), Some(4));
        assert_eq!(node.leaf_indices(), None);
        let (left, right) = node.children().unwrap();
        assert_eq!(left.leaf_indices(), Some(&[2][..]));
        assert_eq!(right.leaf_indices(), Some(&[7][..]));
    }

    #[test]
    #[should_panic(expected = "at least one training index")]
    fn leaves_reject_empty_index_storage() {
        KdTreeNode::leaf(Vec::new());
    }

    #[test]
    #[should_panic(expected = "must be unique")]
    fn leaves_reject_duplicate_training_indices() {
        KdTreeNode::leaf(vec![3, 3]);
    }
}
