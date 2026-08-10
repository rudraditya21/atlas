use crate::{AtlasNdResult, NDArray, Numeric, OperandMetadata, ShapeArg};

impl<T: Numeric> NDArray<T> {
    /// Creates a dense row-major array filled with `value`.
    pub fn new<S>(shape: S, value: T) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        Self::full(shape, value)
    }

    /// Creates a dense row-major array through the dedicated empty-construction path.
    pub fn empty<S>(shape: S) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        let shape = shape.into_shape_vec();
        let size = crate::checked_element_count(&shape)?;
        let mut data = Vec::with_capacity(size);

        // Safe NDArray<T> access cannot expose truly uninitialized elements.
        data.resize_with(size, T::zero);

        Self::from_row_major_parts(shape, data)
    }

    /// Creates a dense row-major array filled with `value`.
    pub fn full<S>(shape: S, value: T) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        let shape = shape.into_shape_vec();
        let size = crate::checked_element_count(&shape)?;

        Self::from_row_major_parts(shape, vec![value; size])
    }

    /// Creates a dense row-major array filled with zeros.
    pub fn zeros<S>(shape: S) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        Self::full(shape, T::zero())
    }

    /// Creates a dense row-major array of zeros with the same logical shape and dtype.
    pub fn zeros_like<O>(other: &O) -> AtlasNdResult<Self>
    where
        O: OperandMetadata<T> + ?Sized,
    {
        Self::zeros(other.shape())
    }

    /// Creates a dense row-major array filled with ones.
    pub fn ones<S>(shape: S) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        Self::full(shape, T::one())
    }

    /// Creates a dense row-major array of ones with the same logical shape and dtype.
    pub fn ones_like<O>(other: &O) -> AtlasNdResult<Self>
    where
        O: OperandMetadata<T> + ?Sized,
    {
        Self::ones(other.shape())
    }

    /// Creates a square identity matrix with ones on the main diagonal.
    pub fn eye(size: usize) -> AtlasNdResult<Self> {
        let shape = vec![size, size];
        let element_count = crate::checked_element_count(&shape)?;
        let mut data = vec![T::zero(); element_count];

        for index in 0..size {
            data[index * size + index] = T::one();
        }

        Self::from_row_major_parts(shape, data)
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_shape_vec<S>(shape: S, data: Vec<T>) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        Self::from_row_major_parts(shape.into_shape_vec(), data)
    }

    /// Creates a dense row-major array from an explicit shape and backing data.
    pub fn from_vec<S>(shape: S, data: Vec<T>) -> AtlasNdResult<Self>
    where
        S: ShapeArg,
    {
        Self::from_shape_vec(shape, data)
    }

    /// Creates a dense row-major array filled with `value` and the same logical shape and dtype.
    pub fn full_like<O>(other: &O, value: T) -> AtlasNdResult<Self>
    where
        O: OperandMetadata<T> + ?Sized,
    {
        Self::full(other.shape(), value)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn new_builds_a_contiguous_row_major_array() {
        let array = NDArray::new([2, 3], 5_i32).unwrap();

        assert_eq!(array.len(), 6);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.strides(), &[3, 1]);
        assert!(array.is_contiguous());
        assert_eq!(array.data(), &[5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn empty_builds_a_contiguous_row_major_array() {
        let array = NDArray::<i32>::empty([2, 3]).unwrap();

        assert_eq!(array.len(), 6);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.strides(), &[3, 1]);
        assert!(array.is_contiguous());
        assert_eq!(array.data(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn constructors_handle_scalar_shapes_consistently() {
        let full = NDArray::full([], 7_i32).unwrap();
        let empty = NDArray::<i32>::empty([]).unwrap();
        let zeros = NDArray::<i32>::zeros([]).unwrap();
        let ones = NDArray::<i32>::ones([]).unwrap();
        let from_shape_vec = NDArray::from_shape_vec([], vec![11_i32]).unwrap();

        assert_eq!(full.shape(), &[] as &[usize]);
        assert_eq!(full.strides(), &[] as &[usize]);
        assert_eq!(full.len(), 1);
        assert_eq!(full.data(), &[7]);

        assert_eq!(empty.shape(), &[] as &[usize]);
        assert_eq!(empty.strides(), &[] as &[usize]);
        assert_eq!(empty.len(), 1);
        assert_eq!(empty.data(), &[0]);

        assert_eq!(zeros.shape(), &[] as &[usize]);
        assert_eq!(zeros.data(), &[0]);
        assert_eq!(ones.shape(), &[] as &[usize]);
        assert_eq!(ones.data(), &[1]);

        assert_eq!(from_shape_vec.shape(), &[] as &[usize]);
        assert_eq!(from_shape_vec.data(), &[11]);
        assert!(full.is_contiguous());
        assert!(empty.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn constructors_handle_empty_one_dimensional_shapes_consistently() {
        let empty = NDArray::<i32>::empty([0]).unwrap();
        let full = NDArray::full([0], 7_i32).unwrap();
        let zeros = NDArray::<i32>::zeros([0]).unwrap();
        let ones = NDArray::<i32>::ones([0]).unwrap();
        let from_shape_vec = NDArray::<i32>::from_shape_vec([0], Vec::new()).unwrap();

        assert_eq!(empty.shape(), &[0]);
        assert_eq!(full.shape(), &[0]);
        assert_eq!(zeros.shape(), &[0]);
        assert_eq!(ones.shape(), &[0]);
        assert_eq!(from_shape_vec.shape(), &[0]);
        assert_eq!(empty.strides(), &[1]);
        assert_eq!(full.strides(), &[1]);
        assert_eq!(zeros.strides(), &[1]);
        assert_eq!(ones.strides(), &[1]);
        assert_eq!(from_shape_vec.strides(), &[1]);
        assert!(empty.data().is_empty());
        assert!(full.data().is_empty());
        assert!(zeros.data().is_empty());
        assert!(ones.data().is_empty());
        assert!(from_shape_vec.data().is_empty());
        assert!(empty.is_contiguous());
        assert!(full.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn constructors_handle_zero_sized_dimensions() {
        let empty = NDArray::<i32>::empty([2, 0, 3]).unwrap();
        let full = NDArray::full([2, 0, 3], 9_i32).unwrap();
        let zeros = NDArray::<i32>::zeros([0, 4]).unwrap();
        let from_shape_vec = NDArray::<i32>::from_shape_vec([0, 2], Vec::new()).unwrap();

        assert_eq!(empty.shape(), &[2, 0, 3]);
        assert_eq!(empty.len(), 0);
        assert!(empty.data().is_empty());
        assert_eq!(empty.strides(), &[0, 3, 1]);

        assert_eq!(full.shape(), &[2, 0, 3]);
        assert_eq!(full.len(), 0);
        assert!(full.data().is_empty());
        assert_eq!(full.strides(), &[0, 3, 1]);

        assert_eq!(zeros.shape(), &[0, 4]);
        assert_eq!(zeros.len(), 0);
        assert!(zeros.data().is_empty());
        assert_eq!(from_shape_vec.shape(), &[0, 2]);
        assert_eq!(from_shape_vec.len(), 0);
        assert!(empty.is_contiguous());
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
        let empty = NDArray::<i32>::empty([2, 2]).unwrap();
        let full = NDArray::full([2, 2], 9_i32).unwrap();
        let zeros = NDArray::<i32>::zeros([2, 2]).unwrap();
        let ones = NDArray::<i32>::ones([2, 2]).unwrap();
        let from_shape_vec = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(empty.data(), &[0, 0, 0, 0]);
        assert_eq!(full.data(), &[9, 9, 9, 9]);
        assert_eq!(zeros.data(), &[0, 0, 0, 0]);
        assert_eq!(ones.data(), &[1, 1, 1, 1]);
        assert_eq!(from_shape_vec.data(), &[1, 2, 3, 4]);
        assert!(empty.is_contiguous());
        assert!(full.is_contiguous());
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(from_shape_vec.is_contiguous());
    }

    #[test]
    fn like_constructors_preserve_owned_array_shape_and_dtype() {
        let source = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let zeros = NDArray::zeros_like(&source).unwrap();
        let ones = NDArray::ones_like(&source).unwrap();
        let full = NDArray::full_like(&source, 9_i32).unwrap();

        assert_eq!(zeros.shape(), source.shape());
        assert_eq!(ones.shape(), source.shape());
        assert_eq!(full.shape(), source.shape());
        assert_eq!(zeros.dtype(), source.dtype());
        assert_eq!(ones.dtype(), source.dtype());
        assert_eq!(full.dtype(), source.dtype());
        assert_eq!(zeros.strides(), &[3, 1]);
        assert_eq!(ones.strides(), &[3, 1]);
        assert_eq!(full.strides(), &[3, 1]);
        assert_eq!(zeros.data(), &[0, 0, 0, 0, 0, 0]);
        assert_eq!(ones.data(), &[1, 1, 1, 1, 1, 1]);
        assert_eq!(full.data(), &[9, 9, 9, 9, 9, 9]);
    }

    #[test]
    fn like_constructors_preserve_view_shape_but_return_owned_row_major_arrays() {
        let source = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let view = source.view().transpose();
        let zeros = NDArray::zeros_like(&view).unwrap();
        let ones = NDArray::ones_like(&view).unwrap();
        let full = NDArray::full_like(&view, 7_i32).unwrap();

        assert_eq!(zeros.shape(), view.shape());
        assert_eq!(ones.shape(), view.shape());
        assert_eq!(full.shape(), view.shape());
        assert_eq!(zeros.dtype(), view.dtype());
        assert_eq!(ones.dtype(), view.dtype());
        assert_eq!(full.dtype(), view.dtype());
        assert_eq!(zeros.strides(), &[2, 1]);
        assert_eq!(ones.strides(), &[2, 1]);
        assert_eq!(full.strides(), &[2, 1]);
        assert!(zeros.is_contiguous());
        assert!(ones.is_contiguous());
        assert!(full.is_contiguous());
        assert_eq!(zeros.data(), &[0, 0, 0, 0, 0, 0]);
        assert_eq!(ones.data(), &[1, 1, 1, 1, 1, 1]);
        assert_eq!(full.data(), &[7, 7, 7, 7, 7, 7]);
    }

    #[test]
    fn like_constructors_preserve_scalar_and_zero_sized_shapes() {
        let scalar = NDArray::from_shape_vec([], vec![5_i32]).unwrap();
        let empty_1d = NDArray::<i32>::from_shape_vec([0], Vec::new()).unwrap();
        let zero_sized = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();

        let scalar_zeros = NDArray::zeros_like(&scalar).unwrap();
        let scalar_ones = NDArray::ones_like(&scalar).unwrap();
        let scalar_full = NDArray::full_like(&scalar, 8_i32).unwrap();
        let empty_1d_zeros = NDArray::zeros_like(&empty_1d).unwrap();
        let empty_1d_full = NDArray::full_like(&empty_1d, 8_i32).unwrap();
        let zero_sized_zeros = NDArray::zeros_like(&zero_sized).unwrap();
        let zero_sized_full = NDArray::full_like(&zero_sized, 8_i32).unwrap();

        assert_eq!(scalar_zeros.shape(), &[] as &[usize]);
        assert_eq!(scalar_ones.shape(), &[] as &[usize]);
        assert_eq!(scalar_full.shape(), &[] as &[usize]);
        assert_eq!(scalar_zeros.data(), &[0]);
        assert_eq!(scalar_ones.data(), &[1]);
        assert_eq!(scalar_full.data(), &[8]);

        assert_eq!(empty_1d_zeros.shape(), &[0]);
        assert_eq!(empty_1d_full.shape(), &[0]);
        assert_eq!(empty_1d_zeros.strides(), &[1]);
        assert_eq!(empty_1d_full.strides(), &[1]);
        assert!(empty_1d_zeros.data().is_empty());
        assert!(empty_1d_full.data().is_empty());

        assert_eq!(zero_sized_zeros.shape(), &[2, 0, 3]);
        assert_eq!(zero_sized_full.shape(), &[2, 0, 3]);
        assert_eq!(zero_sized_zeros.strides(), &[0, 3, 1]);
        assert_eq!(zero_sized_full.strides(), &[0, 3, 1]);
        assert!(zero_sized_zeros.data().is_empty());
        assert!(zero_sized_full.data().is_empty());
    }

    #[test]
    fn eye_creates_a_contiguous_identity_matrix() {
        let identity = NDArray::<i32>::eye(3).unwrap();

        assert_eq!(identity.shape(), &[3, 3]);
        assert_eq!(identity.strides(), &[3, 1]);
        assert_eq!(identity.data(), &[1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert!(identity.is_contiguous());
    }

    #[test]
    fn eye_supports_zero_sized_identity() {
        let identity = NDArray::<i32>::eye(0).unwrap();

        assert_eq!(identity.shape(), &[0, 0]);
        assert_eq!(identity.strides(), &[0, 1]);
        assert!(identity.data().is_empty());
        assert!(identity.is_contiguous());
    }

    #[test]
    fn constructors_accept_slice_like_shape_arguments() {
        let dynamic_shape = vec![2, 3];

        let from_slice = NDArray::<i32>::zeros(dynamic_shape.as_slice()).unwrap();
        let from_array_ref = NDArray::<i32>::ones([2, 3]).unwrap();

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
    fn full_reports_shape_overflow_explicitly() {
        assert_eq!(
            NDArray::<i32>::full([usize::MAX, 2], 0).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
        );
    }

    #[test]
    fn empty_reports_shape_overflow_explicitly() {
        assert_eq!(
            NDArray::<i32>::empty([usize::MAX, 2]).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
        );
    }

    #[test]
    fn eye_reports_shape_overflow_explicitly() {
        assert_eq!(
            NDArray::<i32>::eye(usize::MAX).unwrap_err(),
            AtlasNdError::ShapeOverflow {
                op: "element count",
                shape: vec![usize::MAX, usize::MAX]
            }
        );
    }
}
