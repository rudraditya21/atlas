use super::{array::NDArray, traits::Numeric};

impl<T: Numeric> NDArray<T> {
    fn offset(&self, indices: &[usize]) -> usize {
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

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
