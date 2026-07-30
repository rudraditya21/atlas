use super::{traits::Numeric, view::ArrayView};

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn slice(&self, starts: &[usize], new_shape: Vec<usize>) -> ArrayView<'a, T> {
        assert_eq!(starts.len(), self.shape.len());

        let mut offset = self.offset;

        for ((start, dim), stride) in starts
            .iter()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
        {
            assert!(*start < *dim);
            offset += start * stride;
        }

        ArrayView {
            data: self.data,
            offset,
            shape: new_shape,
            strides: self.strides.clone(),
        }
    }
}
