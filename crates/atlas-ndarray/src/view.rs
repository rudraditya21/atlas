use super::{array::NDArray, traits::Numeric};

#[derive(Debug, Clone)]
pub struct ArrayView<'a, T: Numeric> {
    pub(crate) data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: Numeric> NDArray<T> {
    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView {
            data: &self.data,
            offset: 0,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    fn offset(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.shape.len());

        let mut offset = self.offset;

        for ((i, dim), stride) in index.iter().zip(self.shape.iter()).zip(self.strides.iter()) {
            assert!(*i < *dim, "Index out of bounds");
            offset += i * stride;
        }

        offset
    }

    pub fn get(&self, index: &[usize]) -> &T {
        let idx = self.offset(index);
        &self.data[idx]
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
