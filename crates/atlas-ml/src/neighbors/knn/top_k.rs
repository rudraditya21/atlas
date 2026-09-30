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

pub(crate) struct BoundedNeighborSet {
    capacity: usize,
    neighbors: Vec<Neighbor>,
}

impl BoundedNeighborSet {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0);

        Self { capacity, neighbors: Vec::with_capacity(capacity) }
    }

    pub(crate) fn insert(&mut self, index: usize, distance: f64) {
        let neighbor = Neighbor { index, distance };
        let position = self.neighbors.binary_search(&neighbor).unwrap_or_else(|position| position);
        if position >= self.capacity {
            return;
        }

        self.neighbors.insert(position, neighbor);
        if self.neighbors.len() > self.capacity {
            self.neighbors.pop();
        }
    }

    pub(crate) fn insert_distance_block<I>(&mut self, training_start: usize, distances: I)
    where
        I: IntoIterator<Item = f64>,
    {
        for (offset, distance) in distances.into_iter().enumerate() {
            self.insert(training_start + offset, distance);
        }
    }

    pub(crate) fn is_full(&self) -> bool {
        self.neighbors.len() == self.capacity
    }

    pub(crate) fn neighbors(&self) -> &[Neighbor] {
        &self.neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundedNeighborSet, Neighbor};

    fn neighbor(index: usize, distance: f64) -> Neighbor {
        Neighbor { index, distance }
    }

    #[test]
    fn neighbors_sort_by_distance_then_training_index() {
        let mut neighbors = vec![neighbor(3, 1.0), neighbor(1, 1.0), neighbor(2, 0.5)];

        neighbors.sort();

        assert_eq!(neighbors, vec![neighbor(2, 0.5), neighbor(1, 1.0), neighbor(3, 1.0)]);
    }

    #[test]
    fn inserts_neighbors_in_sorted_order() {
        let mut set = BoundedNeighborSet::new(3);

        set.insert(3, 3.0);
        set.insert(1, 1.0);
        set.insert(2, 2.0);

        assert_eq!(set.neighbors(), &[neighbor(1, 1.0), neighbor(2, 2.0), neighbor(3, 3.0)]);
    }

    #[test]
    fn replaces_the_farthest_neighbor_when_full() {
        let mut set = BoundedNeighborSet::new(2);

        set.insert_distance_block(1, [1.0, 2.0, 3.0, 4.0]);

        assert!(set.is_full());
        assert_eq!(set.neighbors(), &[neighbor(1, 1.0), neighbor(2, 2.0)]);
    }

    #[test]
    fn retains_lower_training_indices_for_distance_ties() {
        let mut set = BoundedNeighborSet::new(2);

        set.insert(4, 1.0);
        set.insert(2, 1.0);
        set.insert(3, 1.0);

        assert_eq!(set.neighbors(), &[neighbor(2, 1.0), neighbor(3, 1.0)]);
    }

    #[test]
    fn retains_only_the_nearest_neighbor_for_k_one() {
        let mut set = BoundedNeighborSet::new(1);

        set.insert(2, 2.0);
        set.insert(1, 1.0);

        assert!(set.is_full());
        assert_eq!(set.neighbors(), &[neighbor(1, 1.0)]);
    }
}
