use super::{traits::Numeric, view::ArrayView};

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn transpose(mut self) -> Self {
        self.shape.reverse();
        self.strides.reverse();

        self
    }
}

#[cfg(test)]
mod tests {
    use crate::array::NDArray;

    #[test]
    fn transpose_reverses_shape_and_strides() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let transposed = array.view().transpose();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert_eq!(*transposed.get(&[2, 1]).unwrap(), 5);
    }

    #[test]
    fn transpose_is_not_marked_contiguous_when_layout_becomes_strided() {
        let array = NDArray::new(vec![2, 3], 1_i32);
        let transposed = array.view().transpose();

        assert!(!transposed.is_contiguous());
    }
}
