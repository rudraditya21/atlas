use super::traits::Numeric;
use crate::{
    AtlasNdError, AtlasNdResult, CastMode, DType, RuntimeDType, RuntimeScalar,
    internal::{
        layout::{dense_storage_slice, is_contiguous_layout},
        shape::checked_row_major_metadata,
        validate_owned_array_invariants,
    },
};

#[derive(Clone, Debug)]
pub struct NDArray<T: Numeric> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: Numeric> NDArray<T> {
    pub(crate) fn from_row_major_parts(shape: Vec<usize>, data: Vec<T>) -> AtlasNdResult<Self> {
        let (expected_len, strides) = checked_row_major_metadata(&shape)?;
        if data.len() != expected_len {
            return Err(AtlasNdError::ShapeMismatch { expected: expected_len, actual: data.len() });
        }

        let array = Self { data, shape, strides };
        array.validate_invariants()?;

        Ok(array)
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype(&self) -> DType
    where
        T: RuntimeDType,
    {
        DType::of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_contiguous(&self) -> bool {
        is_contiguous_layout(&self.shape, &self.strides)
    }

    pub fn dense_slice(&self) -> &[T] {
        dense_storage_slice(&self.data, 0, &self.shape, &self.strides)
            .expect("owned arrays always expose a dense storage region")
    }

    pub fn astype<U>(&self) -> AtlasNdResult<NDArray<U>>
    where
        T: RuntimeScalar,
        U: Numeric + RuntimeScalar,
    {
        self.astype_with_mode(CastMode::Checked)
    }

    pub fn astype_with_mode<U>(&self, mode: CastMode) -> AtlasNdResult<NDArray<U>>
    where
        T: RuntimeScalar,
        U: Numeric + RuntimeScalar,
    {
        let from = self.dtype();
        let to = U::dtype();
        let mut casted = Vec::with_capacity(self.data.len());

        for &value in &self.data {
            let casted_value = value
                .into_scalar_value()
                .cast(to, mode)
                .and_then(U::from_scalar_value)
                .ok_or(AtlasNdError::InvalidCast { from, to, mode })?;
            casted.push(casted_value);
        }

        NDArray::from_row_major_parts(self.shape.clone(), casted)
    }

    pub(crate) fn validate_invariants(&self) -> AtlasNdResult<()> {
        validate_owned_array_invariants(self.data.len(), &self.shape, &self.strides)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, CastMode, DType, NDArray};

    #[test]
    fn scalar_arrays_are_contiguous() {
        let array = NDArray::new(vec![], 7_i32).unwrap();

        assert_eq!(array.len(), 1);
        assert_eq!(array.ndim(), 0);
        assert!(array.is_contiguous());
        assert_eq!(array.strides(), &[] as &[usize]);
        assert_eq!(array.dense_slice(), &[7]);
        assert_eq!(array.validate_invariants(), Ok(()));
    }

    #[test]
    fn dense_slice_preserves_owned_storage_for_empty_and_multidimensional_arrays() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

        assert_eq!(matrix.dense_slice(), matrix.data());
        assert_eq!(empty.dense_slice(), &[] as &[i32]);
    }

    #[test]
    fn from_row_major_parts_reuses_owned_boundary_checks() {
        assert_eq!(
            NDArray::<i32>::from_row_major_parts(vec![2, 2], vec![1, 2, 3]).unwrap_err(),
            AtlasNdError::ShapeMismatch { expected: 4, actual: 3 }
        );
    }

    #[test]
    fn arrays_expose_runtime_dtype_information() {
        let ints = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();
        let floats = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert_eq!(ints.dtype(), DType::I32);
        assert_eq!(floats.dtype(), DType::F64);
    }

    #[test]
    fn astype_converts_arrays_between_supported_numeric_dtypes() {
        let ints = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
        let floats = ints.astype::<f64>().unwrap();

        assert_eq!(floats.shape(), &[2, 2]);
        assert_eq!(floats.strides(), &[2, 1]);
        assert_eq!(floats.dtype(), DType::F64);
        assert_eq!(floats.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn astype_with_checked_mode_rejects_lossy_conversions() {
        let floats = NDArray::from_shape_vec([2], vec![1.25_f64, 2.75]).unwrap();

        assert_eq!(
            floats.astype_with_mode::<i32>(CastMode::Checked).unwrap_err(),
            AtlasNdError::InvalidCast { from: DType::F64, to: DType::I32, mode: CastMode::Checked }
        );
    }

    #[test]
    fn astype_rejects_lossy_conversions_by_default() {
        let floats = NDArray::from_shape_vec([3], vec![1.25_f64, 2.75, -3.5]).unwrap();

        assert_eq!(
            floats.astype::<i32>().unwrap_err(),
            AtlasNdError::InvalidCast { from: DType::F64, to: DType::I32, mode: CastMode::Checked }
        );
    }

    #[test]
    fn astype_with_lossy_mode_allows_explicit_narrowing_and_truncation() {
        let floats = NDArray::from_shape_vec([3], vec![1.25_f64, 2.75, -3.5]).unwrap();
        let ints = floats.astype_with_mode::<i32>(CastMode::Lossy).unwrap();

        assert_eq!(ints.dtype(), DType::I32);
        assert_eq!(ints.data(), &[1, 2, -3]);
    }
}
