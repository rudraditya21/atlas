use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbor {
    pub(crate) index: usize,
    pub(crate) distance: f64,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance).then_with(|| self.index.cmp(&other.index))
    }
}

#[cfg(test)]
mod tests {
    use super::Neighbor;

    #[test]
    fn neighbors_sort_by_distance_then_training_index() {
        let mut neighbors = vec![
            Neighbor { index: 3, distance: 1.0 },
            Neighbor { index: 1, distance: 1.0 },
            Neighbor { index: 2, distance: 0.5 },
        ];

        neighbors.sort();

        assert_eq!(
            neighbors,
            vec![
                Neighbor { index: 2, distance: 0.5 },
                Neighbor { index: 1, distance: 1.0 },
                Neighbor { index: 3, distance: 1.0 },
            ]
        );
    }
}
