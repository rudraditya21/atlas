use super::{array::NDArray, stride::compute_strides, traits::Numeric};

impl<T: Numeric> NDArray<T> {
    pub fn new(shape: Vec<usize>, value: T) -> Self {
        let size = shape.iter().product();

        Self {
            data: vec![value; size],
            strides: compute_strides(&shape),
            shape,
        }
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
        let expected: usize = shape.iter().product();

        assert_eq!(expected, data.len(), "Shape does not match data length");

        Self {
            data,
            strides: compute_strides(&shape),
            shape,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
