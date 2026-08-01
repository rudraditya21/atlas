use super::{array::NDArray, traits::Numeric};

impl<T: Numeric> NDArray<T> {
    fn offset(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(self.shape.len(), self.strides.len());
        assert_eq!(indices.len(), self.shape.len(), "Dimension mismatch");

        let mut offset = 0;
        for ((index, dim), stride) in indices
            .iter()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
        {
            assert!(*index < *dim, "Index out of bounds");
            offset += index * stride;
        }

        offset
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        let idx = self.offset(indices);
        &self.data[idx]
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.offset(indices);
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::NDArray;

    #[test]
    fn get_uses_row_major_offsets_for_two_dimensional_arrays() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]);

        assert_eq!(*array.get(&[0, 0]), 0);
        assert_eq!(*array.get(&[0, 1]), 1);
        assert_eq!(*array.get(&[1, 0]), 3);
        assert_eq!(*array.get(&[1, 2]), 5);
    }

    #[test]
    fn get_uses_row_major_offsets_for_three_dimensional_arrays() {
        let array = NDArray::from_vec(vec![2, 3, 4], (0_i32..24).collect());

        assert_eq!(*array.get(&[0, 0, 0]), 0);
        assert_eq!(*array.get(&[0, 1, 2]), 6);
        assert_eq!(*array.get(&[1, 0, 0]), 12);
        assert_eq!(*array.get(&[1, 2, 3]), 23);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn get_rejects_wrong_dimension_count() {
        let array = NDArray::new(vec![2, 3], 0_i32);

        let _ = array.get(&[0]);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn get_rejects_out_of_bounds_indices() {
        let array = NDArray::new(vec![2, 3], 0_i32);

        let _ = array.get(&[2, 0]);
    }

    #[test]
    fn get_mut_updates_the_underlying_element() {
        let mut array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]);

        *array.get_mut(&[1, 0]) = 9;

        assert_eq!(array.data(), &[1, 2, 9, 4]);
    }
}
