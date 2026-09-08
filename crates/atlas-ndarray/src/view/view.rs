use crate::{
    ArrayElement, AsArray, AtlasNdResult, AxisIndex, CastMode, DType, NDArray, ReductionOp,
    RuntimeDType, RuntimeScalar,
    core::{asarray::cast_array, axis::normalize_and_offset_indices},
    internal::{
        layout::{dense_storage_slice, is_contiguous_layout},
        materialize_contiguous_array,
        shape::element_count,
        validate_view_invariants,
    },
};

#[derive(Debug, Clone)]
pub struct ArrayView<'a, T: ArrayElement> {
    pub(crate) data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: ArrayElement> NDArray<T> {
    /// Returns a metadata-only borrow of this array.
    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView::from_parts(&self.data, 0, self.shape.clone(), self.strides.clone())
            .expect("owned arrays always expose valid view metadata")
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    const ITEM_OP: &'static str = "item";

    pub(crate) fn from_parts(
        data: &'a [T],
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> AtlasNdResult<Self> {
        let view = Self { data, offset, shape, strides };
        view.validate_invariants()?;
        Ok(view)
    }

    fn offset_for_index<I: AxisIndex>(&self, index: &[I]) -> AtlasNdResult<usize> {
        normalize_and_offset_indices(self.offset, index, &self.shape, &self.strides)
    }

    pub fn get<I: AxisIndex>(&self, index: &[I]) -> AtlasNdResult<&T> {
        let idx = self.offset_for_index(index)?;
        Ok(&self.data[idx])
    }

    pub fn item(&self) -> AtlasNdResult<T> {
        if self.len() != 1 {
            return Err(crate::AtlasNdError::InvalidArgument {
                op: Self::ITEM_OP,
                reason: "array must contain exactly one element",
            });
        }

        Ok(self.data[self.offset])
    }

    pub fn item_at<I: AxisIndex>(&self, index: &[I]) -> AtlasNdResult<T> {
        Ok(*self.get(index)?)
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        element_count(&self.shape)
    }

    /// Returns the total number of logical elements in the view.
    pub fn size(&self) -> usize {
        self.len()
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

    pub fn reduction_result_dtype(&self, op: ReductionOp) -> Option<DType>
    where
        T: RuntimeDType,
    {
        self.dtype().reduction_result_dtype(op)
    }

    pub fn reduction_accumulator_dtype(&self, op: ReductionOp) -> Option<DType>
    where
        T: RuntimeDType,
    {
        self.dtype().reduction_accumulator_dtype(op)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `false` because an `ArrayView` borrows its backing storage.
    pub const fn is_owned(&self) -> bool {
        false
    }

    pub fn is_contiguous(&self) -> bool {
        is_contiguous_layout(&self.shape, &self.strides)
    }

    pub fn dense_slice(&self) -> Option<&'a [T]> {
        dense_storage_slice(self.data, self.offset, &self.shape, &self.strides)
    }

    /// Borrows this view without materializing it.
    pub fn asarray(&self) -> AsArray<'a, T> {
        AsArray::Borrowed(self.clone())
    }

    /// Materializes this view into an owned contiguous array.
    pub fn to_owned(&self) -> NDArray<T> {
        materialize_contiguous_array(self)
    }

    /// Materializes this view into an owned contiguous array.
    pub fn copy(&self) -> NDArray<T> {
        self.to_owned()
    }

    /// Casts logical view values into a new owned array, rejecting lossy conversions.
    pub fn astype<U>(&self) -> AtlasNdResult<NDArray<U>>
    where
        T: RuntimeScalar,
        U: ArrayElement + RuntimeScalar,
    {
        self.astype_with_mode(CastMode::Checked)
    }

    /// Casts logical view values into a new owned array using the selected conversion mode.
    pub fn astype_with_mode<U>(&self, mode: CastMode) -> AtlasNdResult<NDArray<U>>
    where
        T: RuntimeScalar,
        U: ArrayElement + RuntimeScalar,
    {
        cast_array(self.shape.clone(), self.iter().copied(), mode)
    }

    pub(crate) fn validate_invariants(&self) -> AtlasNdResult<()> {
        validate_view_invariants(self.data.len(), self.offset, &self.shape, &self.strides)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, DType, NDArray, ReductionOp};

    #[test]
    fn view_preserves_owned_layout_metadata() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();

        assert_eq!(view.data(), array.data());
        assert_eq!(view.offset(), 0);
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view.len(), 6);
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.dtype(), DType::I32);
        assert!(view.is_contiguous());
        assert!(view.dense_slice().is_some());
        assert_eq!(view.dense_slice().unwrap(), array.data());
        assert_eq!(view.validate_invariants(), Ok(()));
    }

    #[test]
    fn views_expose_runtime_reduction_dtype_rules() {
        let array = NDArray::from_shape_vec([2, 2], vec![1_u8, 2, 3, 4]).unwrap();
        let bools = NDArray::from_shape_vec([2], vec![true, false]).unwrap();
        let view = array.view().transpose();
        let bool_view = bools.view();

        assert_eq!(view.reduction_result_dtype(ReductionOp::Sum), Some(DType::Usize));
        assert_eq!(view.reduction_result_dtype(ReductionOp::Min), Some(DType::U8));
        assert_eq!(view.reduction_result_dtype(ReductionOp::Mean), Some(DType::F64));
        assert_eq!(bool_view.reduction_result_dtype(ReductionOp::Any), Some(DType::Bool));
        assert_eq!(view.reduction_result_dtype(ReductionOp::All), None);
    }

    #[test]
    fn view_get_uses_underlying_array_storage() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();

        assert_eq!(*view.get(&[1, 2]).unwrap(), 5);
    }

    #[test]
    fn view_get_rejects_dimension_mismatch() {
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();
        let view = array.view();

        let error = view.get(&[0]).unwrap_err();

        assert_eq!(error, AtlasNdError::DimensionMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn scalar_view_uses_empty_index_and_reports_rank_mismatch_consistently() {
        let array = NDArray::new([], 13_i32).unwrap();
        let view = array.view();

        assert_eq!(*view.get(&[] as &[i64]).unwrap(), 13);
        assert_eq!(view.item().unwrap(), 13);
        assert_eq!(view.item_at(&[] as &[i64]).unwrap(), 13);
        assert_eq!(
            view.get(&[0]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 0, actual: 1 }
        );
    }

    #[test]
    fn item_supports_single_element_non_scalar_views_and_rejects_larger_inputs() {
        let array = NDArray::from_vec([1, 1], vec![7_i32]).unwrap();
        let single = array.view();
        let larger = NDArray::from_vec([1, 2], vec![7_i32, 8]).unwrap();

        assert_eq!(single.item().unwrap(), 7);
        assert_eq!(
            larger.view().item().unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "item",
                reason: "array must contain exactly one element",
            }
        );
    }

    #[test]
    fn item_at_copies_indexed_values_from_views_with_shared_normalization() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();

        assert_eq!(view.item_at(&[2, 1]).unwrap(), 5);
        assert_eq!(view.item_at(&[-2, 0]).unwrap(), 1);
        assert_eq!(
            view.item_at(&[3, 0]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 0, index: 3, dim: 3 }
        );
    }

    #[test]
    fn view_get_reports_axis_specific_bounds_errors() {
        let array = NDArray::new([2, 3], 0_i32).unwrap();
        let view = array.view();

        assert_eq!(
            view.get(&[0, 3]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 }
        );
    }

    #[test]
    fn view_get_supports_negative_indices_through_shared_normalization() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();

        assert_eq!(*view.get(&[-1, -1]).unwrap(), 5);
        assert_eq!(*view.get(&[-2, 0]).unwrap(), 1);
        assert_eq!(
            view.get(&[-4, 0]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 0, index: -4, dim: 3 }
        );
    }

    #[test]
    fn transposed_view_is_storage_dense_without_being_row_major_contiguous() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();

        assert!(!view.is_contiguous());
        assert!(view.dense_slice().is_some());
        assert_eq!(view.dense_slice().unwrap(), array.data());
        assert_eq!(view.validate_invariants(), Ok(()));
    }

    #[test]
    fn sliced_view_is_not_storage_dense_when_it_skips_elements() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([0, 1], [2, 2]).unwrap();

        assert!(view.dense_slice().is_none());
    }

    #[test]
    fn empty_dense_views_return_valid_empty_slices() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([1, 3], [1, 0]).unwrap();

        assert!(view.dense_slice().is_some());
        assert_eq!(view.dense_slice().unwrap(), &[] as &[i32]);
        assert_eq!(view.validate_invariants(), Ok(()));
    }

    #[test]
    fn to_owned_materializes_views_as_contiguous_owned_arrays() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();
        let owned = view.to_owned();

        assert_eq!(owned.shape(), &[3, 2]);
        assert_eq!(owned.strides(), &[2, 1]);
        assert_eq!(owned.data(), &[0, 3, 1, 4, 2, 5]);
        assert!(owned.is_contiguous());
    }
}
