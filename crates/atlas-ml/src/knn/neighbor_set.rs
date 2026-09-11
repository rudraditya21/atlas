use super::neighbor::Neighbor;

pub(crate) struct BoundedNeighborSet {
    capacity: usize,
    neighbors: Vec<Neighbor>,
}

impl BoundedNeighborSet {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0);

        Self { capacity, neighbors: Vec::with_capacity(capacity) }
    }

    pub(crate) fn insert(&mut self, neighbor: Neighbor) {
        let position = self.neighbors.binary_search(&neighbor).unwrap_or_else(|position| position);
        if position >= self.capacity {
            return;
        }

        self.neighbors.insert(position, neighbor);
        if self.neighbors.len() > self.capacity {
            self.neighbors.pop();
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
    use super::BoundedNeighborSet;
    use crate::knn::neighbor::Neighbor;

    fn neighbor(index: usize, distance: f64) -> Neighbor {
        Neighbor { index, distance }
    }

    #[test]
    fn inserts_neighbors_in_sorted_order() {
        let mut set = BoundedNeighborSet::new(3);

        set.insert(neighbor(3, 3.0));
        set.insert(neighbor(1, 1.0));
        set.insert(neighbor(2, 2.0));

        assert_eq!(set.neighbors(), &[neighbor(1, 1.0), neighbor(2, 2.0), neighbor(3, 3.0)]);
    }

    #[test]
    fn replaces_the_farthest_neighbor_when_full() {
        let mut set = BoundedNeighborSet::new(2);

        set.insert(neighbor(1, 1.0));
        set.insert(neighbor(3, 3.0));
        set.insert(neighbor(2, 2.0));
        set.insert(neighbor(4, 4.0));

        assert!(set.is_full());
        assert_eq!(set.neighbors(), &[neighbor(1, 1.0), neighbor(2, 2.0)]);
    }

    #[test]
    fn retains_lower_training_indices_for_distance_ties() {
        let mut set = BoundedNeighborSet::new(2);

        set.insert(neighbor(4, 1.0));
        set.insert(neighbor(2, 1.0));
        set.insert(neighbor(3, 1.0));

        assert_eq!(set.neighbors(), &[neighbor(2, 1.0), neighbor(3, 1.0)]);
    }

    #[test]
    fn retains_only_the_nearest_neighbor_for_k_one() {
        let mut set = BoundedNeighborSet::new(1);

        set.insert(neighbor(2, 2.0));
        set.insert(neighbor(1, 1.0));

        assert!(set.is_full());
        assert_eq!(set.neighbors(), &[neighbor(1, 1.0)]);
    }
}
