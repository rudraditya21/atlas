use crate::{
    AtlasNdError, AtlasNdResult, NDArray, Numeric, ShapeArg, checked_compute_strides,
    checked_element_count,
};

fn checked_row_major_metadata(shape: &[usize]) -> AtlasNdResult<(usize, Vec<usize>)> {
    let size = checked_element_count(shape)?;
    let strides = checked_compute_strides(shape)?;

    Ok((size, strides))
}

fn validate_owned_boundary<T: Numeric>(array: NDArray<T>, context: &'static str) -> NDArray<T> {
    array
        .validate_invariants()
        .unwrap_or_else(|error| panic!("{context} failed invariant validation: {error}"));
    array
}

impl<T: Numeric> NDArray<T> {
    /// Creates a dense row-major array filled with `value`.
    pub fn new<S>(shape: S, value: T) -> Self
    where
        S: ShapeArg,
    {
        Self::full(shape, value)
    }

    /// Creates a dense row-major array filled with `value`.
    pub fn full<S>(shape: S, value: T) -> Self
    where
        S: ShapeArg,
    {
        let shape = shape.into_shape_vec();
        let (size, strides) = checked_row_major_metadata(&shape).unwrap_or_else(|error| {
            panic!("NDArray::full failed: {error}");
        });

        validate_owned_boundary(Self { data: vec![value; size], strides, shape }, "NDArray::full")
    }

    /// Creates a dense row-major array filled with zeros.
    pub fn zeros<S>(shape: S) -> Self
    where
        S: ShapeArg,
    {
        Self::full(shape, T::zero())
    }

    /// Creates a dense row-major array filled with ones.
    pub fn ones<S>(shape: S) -> Self
    where
        S: ShapeArg,
    {
        Self::full(shape, T::one())
    }

    /// Creates a square identity matrix with ones on the main diagonal.
    pub fn eye(size: usize) -> Self {
        let shape = vec![size, size];
        let (element_count, strides) = checked_row_major_metadata(&shape).unwrap_or_else(|error| {
            panic!("NDArray::eye failed: {error}");
        });
        let mut data = vec![T::zero(); element_count];

        for index in 0..size {
            data[index * size + index] = T::one();
        }

        validate_owned_boundary(Self { data, strides, shape }, "NDArray::eye")
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_shape_vec<S>(shape: S, data: Vec<T>) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        let shape = shape.into_shape_vec();
        let (expected, strides) = checked_row_major_metadata(&shape)?;

        if expected != data.len() {
            return Err(AtlasNdError::ShapeMismatch { expected, actual: data.len() });
        }

        let array = Self { data, strides, shape };
        array.validate_invariants()?;

        Ok(array)
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_vec<S>(shape: S, data: Vec<T>) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        Self::from_shape_vec(shape, data)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn new_builds_a_contiguous_row_major_array() {
        let array = NDArray::new([2, 3], 5_i32);

        assert_eq!(array.len(), 6);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.strides(), &[3, 1]);
        assert!(array.is_contiguous());
        assert_eq!(array.data(), &[5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn constructors_handle_scalar_shapes_consistently() {
        let full = NDArray::full([], 7_i32);
        let zeros = NDArray::<i32>::zeros([]);
        let ones = NDArray::<i32>::ones([]);
        let from_shape_vec = NDArray::from_shape_vec([], vec![11_i32]).unwrap();

        assert_eq!(full.shape(), &[] as &[usize]);
        assert_eq!(full.strides(), &[] as &[usize]);
        assert_eq!(full.len(), 1);
        assert_eq!(full.data(), &[7]);

        assert_eq!(zeros.shape(), &[] as &[usize]);
        assert_eq!(zeros.data(), &[0]);
        assert_eq!(ones.shape(), &[] as &[usize]);
        assert_eq!(ones.data(), &[1]);

        assert_eq!(from_shape_vec.shape(), &[] as &[usize]);
        assert_eq!(from_shape_vec.data(), &[11]);
        assert!(full.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn constructors_handle_zero_sized_dimensions() {
        let full = NDArray::full([2, 0, 3], 9_i32);
        let zeros = NDArray::<i32>::zeros([0, 4]);
        let from_shape_vec = NDArray::<i32>::from_shape_vec([0, 2], Vec::new()).unwrap();

        assert_eq!(full.shape(), &[2, 0, 3]);
        assert_eq!(full.len(), 0);
        assert!(full.data().is_empty());
        assert_eq!(full.strides(), &[0, 3, 1]);

        assert_eq!(zeros.shape(), &[0, 4]);
        assert_eq!(zeros.len(), 0);
        assert!(zeros.data().is_empty());
        assert_eq!(from_shape_vec.shape(), &[0, 2]);
        assert_eq!(from_shape_vec.len(), 0);
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn from_vec_preserves_data_for_contiguous_layout() {
        let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.strides(), &[2, 1]);
        assert!(array.is_contiguous());
        assert_eq!(array.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_vec_rejects_inconsistent_shape_and_data_length() {
        let error = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3]).unwrap_err();

        assert_eq!(error, AtlasNdError::ShapeMismatch { expected: 4, actual: 3 });
    }

    #[test]
    fn from_shape_vec_rejects_invalid_scalar_and_zero_dim_lengths() {
        assert_eq!(
            NDArray::<i32>::from_shape_vec([], Vec::new()).unwrap_err(),
            AtlasNdError::ShapeMismatch { expected: 1, actual: 0 }
        );

        assert_eq!(
            NDArray::<i32>::from_shape_vec([], vec![1_i32, 2]).unwrap_err(),
            AtlasNdError::ShapeMismatch { expected: 1, actual: 2 }
        );

        assert_eq!(
            NDArray::<i32>::from_shape_vec([2, 0], vec![1_i32]).unwrap_err(),
            AtlasNdError::ShapeMismatch { expected: 0, actual: 1 }
        );
    }

    #[test]
    fn full_zeros_ones_and_from_shape_vec_provide_stable_constructor_surface() {
        let full = NDArray::full([2, 2], 9_i32);
        let zeros = NDArray::<i32>::zeros([2, 2]);
        let ones = NDArray::<i32>::ones([2, 2]);
        let from_shape_vec = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(full.data(), &[9, 9, 9, 9]);
        assert_eq!(zeros.data(), &[0, 0, 0, 0]);
        assert_eq!(ones.data(), &[1, 1, 1, 1]);
        assert_eq!(from_shape_vec.data(), &[1, 2, 3, 4]);
        assert!(full.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn eye_creates_a_contiguous_identity_matrix() {
        let identity = NDArray::<i32>::eye(3);

        assert_eq!(identity.shape(), &[3, 3]);
        assert_eq!(identity.strides(), &[3, 1]);
        assert_eq!(identity.data(), &[1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert!(identity.is_contiguous());
    }

    #[test]
    fn eye_supports_zero_sized_identity() {
        let identity = NDArray::<i32>::eye(0);

        assert_eq!(identity.shape(), &[0, 0]);
        assert_eq!(identity.strides(), &[0, 1]);
        assert!(identity.data().is_empty());
        assert!(identity.is_contiguous());
    }

    #[test]
    fn constructors_accept_slice_like_shape_arguments() {
        let dynamic_shape = vec![2, 3];

        let from_slice = NDArray::<i32>::zeros(dynamic_shape.as_slice());
        let from_array_ref = NDArray::<i32>::ones([2, 3]);

        assert_eq!(from_slice.shape(), &[2, 3]);
        assert_eq!(from_array_ref.shape(), &[2, 3]);
    }

    #[test]
    fn from_shape_vec_reports_shape_overflow_explicitly() {
        assert_eq!(
            NDArray::<i32>::from_shape_vec([usize::MAX, 2], Vec::new()).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
        );
    }

    #[test]
    #[should_panic(expected = "NDArray::full failed: shape overflow for element count")]
    fn full_panics_explicitly_on_shape_overflow() {
        let _ = NDArray::<i32>::full([usize::MAX, 2], 0);
    }

    #[test]
    #[should_panic(expected = "NDArray::eye failed: shape overflow for element count")]
    fn eye_panics_explicitly_on_shape_overflow() {
        let _ = NDArray::<i32>::eye(usize::MAX);
    }
}
