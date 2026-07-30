use super::{traits::Numeric, view::ArrayView};

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Self {
        let old_size: usize = self.shape.iter().product();

        let new_size: usize = new_shape.iter().product();

        assert_eq!(
            old_size, new_size,
            "Cannot reshape different number of elements"
        );

        self.shape = new_shape;

        self.strides = super::stride::compute_strides(&self.shape);

        self
    }
}
