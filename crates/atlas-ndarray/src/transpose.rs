use super::{traits::Numeric, view::ArrayView};

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn transpose(mut self) -> Self {
        self.shape.reverse();
        self.strides.reverse();

        self
    }
}
