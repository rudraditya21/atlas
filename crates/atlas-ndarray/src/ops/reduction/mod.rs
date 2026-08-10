mod axis;
mod dispatch;
mod mean;
mod metadata;
mod truth;
mod whole;

use num_traits::ToPrimitive;

use crate::{AtlasNdResult, AxisIndex, NDArray, Numeric, OperandMetadata, view::ArrayView};

use self::{
    axis::{
        max_axis_impl, max_axis_keepdims_impl, mean_axis_impl, mean_axis_keepdims_impl,
        min_axis_impl, min_axis_keepdims_impl, prod_axis_impl, prod_axis_keepdims_impl,
        sum_axis_impl, sum_axis_keepdims_impl,
    },
    truth::{
        all_all, all_axis_impl, all_axis_keepdims_impl, any_all, any_axis_impl,
        any_axis_keepdims_impl,
    },
    whole::{max_all, mean_all, min_all, prod_all, sum_all},
};

impl<T: Numeric> NDArray<T> {
    pub fn sum(&self) -> AtlasNdResult<T> {
        sum_operand(self)
    }

    pub fn prod(&self) -> AtlasNdResult<T> {
        prod_operand(self)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        min_operand(self)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        max_operand(self)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        mean_operand(self)
    }

    pub fn sum_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        sum_axis_operand(self, axis)
    }

    pub fn sum_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        sum_axis_keepdims_operand(self, axis)
    }

    pub fn prod_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        prod_axis_operand(self, axis)
    }

    pub fn prod_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        prod_axis_keepdims_operand(self, axis)
    }

    pub fn min_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        min_axis_operand(self, axis)
    }

    pub fn min_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        min_axis_keepdims_operand(self, axis)
    }

    pub fn max_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        max_axis_operand(self, axis)
    }

    pub fn max_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        max_axis_keepdims_operand(self, axis)
    }

    pub fn mean_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_operand(self, axis)
    }

    pub fn mean_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_keepdims_operand(self, axis)
    }
}

impl NDArray<bool> {
    pub fn all(&self) -> bool {
        all_operand(self)
    }

    pub fn any(&self) -> bool {
        any_operand(self)
    }

    pub fn all_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        all_axis_operand(self, axis)
    }

    pub fn all_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        all_axis_keepdims_operand(self, axis)
    }

    pub fn any_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        any_axis_operand(self, axis)
    }

    pub fn any_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        any_axis_keepdims_operand(self, axis)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn sum(&self) -> AtlasNdResult<T> {
        sum_operand(self)
    }

    pub fn prod(&self) -> AtlasNdResult<T> {
        prod_operand(self)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        min_operand(self)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        max_operand(self)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        mean_operand(self)
    }

    pub fn sum_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        sum_axis_operand(self, axis)
    }

    pub fn sum_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        sum_axis_keepdims_operand(self, axis)
    }

    pub fn prod_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        prod_axis_operand(self, axis)
    }

    pub fn prod_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        prod_axis_keepdims_operand(self, axis)
    }

    pub fn min_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        min_axis_operand(self, axis)
    }

    pub fn min_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        min_axis_keepdims_operand(self, axis)
    }

    pub fn max_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        max_axis_operand(self, axis)
    }

    pub fn max_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        max_axis_keepdims_operand(self, axis)
    }

    pub fn mean_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_operand(self, axis)
    }

    pub fn mean_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_keepdims_operand(self, axis)
    }
}

impl<'a> ArrayView<'a, bool> {
    pub fn all(&self) -> bool {
        all_operand(self)
    }

    pub fn any(&self) -> bool {
        any_operand(self)
    }

    pub fn all_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<bool>> {
        all_axis_operand(self, axis)
    }

    pub fn all_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<bool>> {
        all_axis_keepdims_operand(self, axis)
    }

    pub fn any_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<bool>> {
        any_axis_operand(self, axis)
    }

    pub fn any_axis_keepdims<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<bool>> {
        any_axis_keepdims_operand(self, axis)
    }
}

fn sum_operand<T, O>(operand: &O) -> AtlasNdResult<T>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
{
    sum_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn prod_operand<T, O>(operand: &O) -> AtlasNdResult<T>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
{
    prod_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn min_operand<T, O>(operand: &O) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
{
    min_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn max_operand<T, O>(operand: &O) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
{
    max_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn mean_operand<T, O>(operand: &O) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
    O: OperandMetadata<T> + ?Sized,
{
    mean_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn all_operand<O>(operand: &O) -> bool
where
    O: OperandMetadata<bool> + ?Sized,
{
    all_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn any_operand<O>(operand: &O) -> bool
where
    O: OperandMetadata<bool> + ?Sized,
{
    any_all(operand.data(), operand.offset(), operand.shape(), operand.strides())
}

fn sum_axis_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    sum_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn sum_axis_keepdims_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    sum_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

fn prod_axis_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    prod_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn prod_axis_keepdims_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    prod_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

fn min_axis_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    min_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn min_axis_keepdims_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    min_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

fn max_axis_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    max_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn max_axis_keepdims_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    max_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

fn mean_axis_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    mean_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn mean_axis_keepdims_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    mean_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

fn all_axis_operand<O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<bool>>
where
    O: OperandMetadata<bool> + ?Sized,
    A: AxisIndex,
{
    all_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn all_axis_keepdims_operand<O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<bool>>
where
    O: OperandMetadata<bool> + ?Sized,
    A: AxisIndex,
{
    all_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

fn any_axis_operand<O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<bool>>
where
    O: OperandMetadata<bool> + ?Sized,
    A: AxisIndex,
{
    any_axis_impl(operand.data(), operand.offset(), operand.shape(), operand.strides(), axis)
}

fn any_axis_keepdims_operand<O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<bool>>
where
    O: OperandMetadata<bool> + ?Sized,
    A: AxisIndex,
{
    any_axis_keepdims_impl(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
        axis,
    )
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn whole_array_reductions_work_for_contiguous_arrays() {
        let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum().unwrap(), 10);
        assert_eq!(array.prod().unwrap(), 24);
        assert_eq!(array.min().unwrap(), 1);
        assert_eq!(array.max().unwrap(), 4);
        assert_eq!(array.mean().unwrap(), 2.5);
    }

    #[test]
    fn whole_array_reductions_work_for_scalar_arrays() {
        let array = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(array.sum().unwrap(), 7);
        assert_eq!(array.prod().unwrap(), 7);
        assert_eq!(array.min().unwrap(), 7);
        assert_eq!(array.max().unwrap(), 7);
        assert_eq!(array.mean().unwrap(), 7.0);
    }

    #[test]
    fn whole_array_truth_reductions_follow_boolean_identities() {
        let array = NDArray::from_shape_vec([2, 2], vec![true, true, false, true]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![true]).unwrap();
        let empty = NDArray::<bool>::new([0, 3], false).unwrap();

        assert!(!array.all());
        assert!(array.any());
        assert!(scalar.all());
        assert!(scalar.any());
        assert!(empty.all());
        assert!(!empty.any());
    }

    #[test]
    fn whole_array_reductions_work_for_non_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([0, 1], vec![2, 2]).unwrap();

        assert_eq!(view.sum().unwrap(), 12);
        assert_eq!(view.prod().unwrap(), 40);
        assert_eq!(view.min().unwrap(), 1);
        assert_eq!(view.max().unwrap(), 5);
        assert_eq!(view.mean().unwrap(), 3.0);
    }

    #[test]
    fn whole_array_floating_reductions_preserve_small_terms_for_strided_views() {
        let array =
            NDArray::from_shape_vec([3, 2], vec![1.0e16_f64, 0.0, 1.0, 0.0, -1.0e16, 0.0]).unwrap();
        let view = array.view().slice([0, 0], [3, 1]).unwrap();

        assert_eq!(view.sum().unwrap(), 1.0);
        assert_eq!(view.mean().unwrap(), 1.0 / 3.0);
    }

    #[test]
    fn whole_array_reductions_work_for_empty_dense_views() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([1, 3], [1, 0]).unwrap();

        assert_eq!(view.sum().unwrap_err(), AtlasNdError::EmptyReduction { op: "sum" });
        assert_eq!(view.prod().unwrap_err(), AtlasNdError::EmptyReduction { op: "prod" });
        assert_eq!(view.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(view.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(view.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn whole_array_truth_reductions_work_for_non_contiguous_views() {
        let array =
            NDArray::from_shape_vec([2, 3], vec![true, false, true, true, true, false]).unwrap();
        let view = array.view().transpose();

        assert!(!view.all());
        assert!(view.any());

        let empty = view.slice([0, 1], [3, 0]).unwrap();
        assert!(empty.all());
        assert!(!empty.any());
    }

    #[test]
    fn whole_array_reduction_entry_points_use_logical_empty_semantics_for_mixed_empty_views() {
        let array = NDArray::<i32>::new([2, 0, 3], 1).unwrap();
        let view = array.view().transpose();

        assert_eq!(view.sum().unwrap_err(), AtlasNdError::EmptyReduction { op: "sum" });
        assert_eq!(view.prod().unwrap_err(), AtlasNdError::EmptyReduction { op: "prod" });
        assert_eq!(view.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(view.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(view.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn sum_and_prod_reject_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7).unwrap();

        assert_eq!(array.sum().unwrap_err(), AtlasNdError::EmptyReduction { op: "sum" });
        assert_eq!(array.prod().unwrap_err(), AtlasNdError::EmptyReduction { op: "prod" });
    }

    #[test]
    fn min_max_and_mean_reject_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7).unwrap();

        assert_eq!(array.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(array.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(array.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn axis_reductions_work_for_contiguous_arrays() {
        let array = NDArray::from_vec(vec![2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

        assert_eq!(array.sum_axis(0).unwrap().data(), &[5, 7, 9]);
        assert_eq!(array.sum_axis(1).unwrap().data(), &[6, 15]);
        assert_eq!(array.prod_axis(0).unwrap().data(), &[4, 10, 18]);
        assert_eq!(array.prod_axis(1).unwrap().data(), &[6, 120]);
        assert_eq!(array.min_axis(0).unwrap().data(), &[1, 2, 3]);
        assert_eq!(array.min_axis(1).unwrap().data(), &[1, 4]);
        assert_eq!(array.max_axis(1).unwrap().data(), &[3, 6]);
        assert_eq!(array.max_axis(0).unwrap().data(), &[4, 5, 6]);
        assert_eq!(array.mean_axis(0).unwrap().data(), &[2.5, 3.5, 4.5]);
        assert_eq!(array.mean_axis(1).unwrap().data(), &[2.0, 5.0]);
    }

    #[test]
    fn axis_truth_reductions_work_for_contiguous_arrays() {
        let array =
            NDArray::from_shape_vec([2, 3], vec![true, true, false, true, false, false]).unwrap();

        assert_eq!(array.all_axis(0).unwrap().data(), &[true, false, false]);
        assert_eq!(array.all_axis(1).unwrap().data(), &[false, false]);
        assert_eq!(array.any_axis(0).unwrap().data(), &[true, true, false]);
        assert_eq!(array.any_axis(1).unwrap().data(), &[true, true]);
    }

    #[test]
    fn axis_reductions_support_keepdims_shapes_for_numeric_and_boolean_outputs() {
        let array = NDArray::from_vec(vec![2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let mask =
            NDArray::from_shape_vec([2, 3], vec![true, true, false, true, false, false]).unwrap();

        assert_eq!(array.sum_axis_keepdims(0).unwrap().shape(), &[1, 3]);
        assert_eq!(array.sum_axis_keepdims(0).unwrap().data(), &[5, 7, 9]);
        assert_eq!(array.mean_axis_keepdims(1).unwrap().shape(), &[2, 1]);
        assert_eq!(array.mean_axis_keepdims(1).unwrap().data(), &[2.0, 5.0]);
        assert_eq!(mask.all_axis_keepdims(1).unwrap().shape(), &[2, 1]);
        assert_eq!(mask.all_axis_keepdims(1).unwrap().data(), &[false, false]);
        assert_eq!(mask.any_axis_keepdims(0).unwrap().shape(), &[1, 3]);
        assert_eq!(mask.any_axis_keepdims(0).unwrap().data(), &[true, true, false]);
    }

    #[test]
    fn axis_reductions_work_for_strided_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();

        assert_eq!(view.sum_axis(-2).unwrap().data(), &[3, 12]);
        assert_eq!(view.sum_axis(-1).unwrap().data(), &[3, 5, 7]);
        assert_eq!(view.prod_axis(-2).unwrap().data(), &[0, 60]);
        assert_eq!(view.min_axis(-1).unwrap().data(), &[0, 1, 2]);
        assert_eq!(view.max_axis(-2).unwrap().data(), &[2, 5]);
        assert_eq!(view.mean_axis(-1).unwrap().data(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn axis_floating_reductions_preserve_small_terms_for_dense_and_strided_lanes() {
        let dense =
            NDArray::from_shape_vec([2, 3], vec![1.0e16_f64, 1.0, -1.0e16, 1.0e16, 1.0, -1.0e16])
                .unwrap();
        let strided_source =
            NDArray::from_shape_vec([3, 2], vec![1.0e16_f64, 1.0e16, 1.0, 1.0, -1.0e16, -1.0e16])
                .unwrap();
        let strided = strided_source.view().transpose();

        assert_eq!(dense.sum_axis(1).unwrap().data(), &[1.0, 1.0]);
        assert_eq!(dense.mean_axis(1).unwrap().data(), &[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(strided.sum_axis(1).unwrap().data(), &[1.0, 1.0]);
        assert_eq!(strided.mean_axis(1).unwrap().data(), &[1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn axis_truth_reductions_work_for_strided_views() {
        let array =
            NDArray::from_shape_vec([2, 3], vec![true, false, true, true, true, false]).unwrap();
        let view = array.view().transpose();

        assert_eq!(view.all_axis(-2).unwrap().data(), &[false, false]);
        assert_eq!(view.any_axis(-1).unwrap().data(), &[true, true, true]);
    }

    #[test]
    fn axis_reductions_validate_axis_bounds() {
        let array = NDArray::new(vec![2, 3], 1_i32).unwrap();

        assert_eq!(array.sum_axis(2).unwrap_err(), AtlasNdError::InvalidAxis { axis: 2, ndim: 2 });
        assert_eq!(
            array.sum_axis(-3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -3, ndim: 2 }
        );
        assert_eq!(array.mean_axis(2).unwrap_err(), AtlasNdError::InvalidAxis { axis: 2, ndim: 2 });
        assert_eq!(
            array.max_axis(-3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -3, ndim: 2 }
        );
    }

    #[test]
    fn axis_reductions_handle_empty_axes_consistently() {
        let array = NDArray::<i32>::new(vec![0, 3], 1).unwrap();

        assert_eq!(array.sum_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "sum" });
        assert_eq!(array.prod_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "prod" });
        assert_eq!(array.min_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(array.max_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(array.mean_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn axis_truth_reductions_use_boolean_identities_for_empty_axes() {
        let array = NDArray::<bool>::new([0, 3], false).unwrap();

        assert_eq!(array.all_axis(0).unwrap().shape(), &[3]);
        assert_eq!(array.all_axis(0).unwrap().data(), &[true, true, true]);
        assert_eq!(array.any_axis(0).unwrap().data(), &[false, false, false]);
    }

    #[test]
    fn axis_reductions_preserve_zero_length_output_shapes_when_lanes_are_empty() {
        let array = NDArray::<i32>::new(vec![2, 0, 3], 1).unwrap();

        assert_eq!(array.sum_axis(0).unwrap().shape(), &[0, 3]);
        assert!(array.sum_axis(0).unwrap().data().is_empty());
        assert_eq!(array.prod_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.prod_axis(2).unwrap().data().is_empty());
        assert_eq!(array.min_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.min_axis(2).unwrap().data().is_empty());
        assert_eq!(array.max_axis(0).unwrap().shape(), &[0, 3]);
        assert!(array.max_axis(0).unwrap().data().is_empty());
        assert_eq!(array.mean_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.mean_axis(2).unwrap().data().is_empty());
    }

    #[test]
    fn axis_truth_reductions_preserve_zero_length_output_shapes_when_outputs_are_empty() {
        let array = NDArray::<bool>::new([2, 0, 3], true).unwrap();

        assert_eq!(array.all_axis(0).unwrap().shape(), &[0, 3]);
        assert!(array.all_axis(0).unwrap().data().is_empty());
        assert_eq!(array.any_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.any_axis(2).unwrap().data().is_empty());
    }

    #[test]
    fn axis_reductions_return_scalar_outputs_for_one_dimensional_inputs() {
        let array = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum_axis(-1).unwrap().shape(), &[] as &[usize]);
        assert_eq!(array.sum_axis(-1).unwrap().data(), &[10]);
        assert_eq!(array.prod_axis(-1).unwrap().data(), &[24]);
        assert_eq!(array.min_axis(-1).unwrap().data(), &[1]);
        assert_eq!(array.max_axis(-1).unwrap().data(), &[4]);
        assert_eq!(array.mean_axis(-1).unwrap().data(), &[2.5]);
    }

    #[test]
    fn axis_truth_reductions_return_scalar_outputs_for_one_dimensional_inputs() {
        let array = NDArray::from_shape_vec([4], vec![true, true, false, true]).unwrap();

        assert_eq!(array.all_axis(-1).unwrap().shape(), &[] as &[usize]);
        assert_eq!(array.all_axis(-1).unwrap().data(), &[false]);
        assert_eq!(array.any_axis(-1).unwrap().data(), &[true]);
    }

    #[test]
    fn axis_keepdims_reductions_preserve_singleton_axes_for_one_dimensional_inputs() {
        let array = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
        let mask = NDArray::from_shape_vec([4], vec![true, true, false, true]).unwrap();

        assert_eq!(array.sum_axis_keepdims(-1).unwrap().shape(), &[1]);
        assert_eq!(array.sum_axis_keepdims(-1).unwrap().data(), &[10]);
        assert_eq!(array.mean_axis_keepdims(-1).unwrap().data(), &[2.5]);
        assert_eq!(mask.all_axis_keepdims(-1).unwrap().shape(), &[1]);
        assert_eq!(mask.all_axis_keepdims(-1).unwrap().data(), &[false]);
        assert_eq!(mask.any_axis_keepdims(-1).unwrap().data(), &[true]);
    }

    #[test]
    fn whole_array_reductions_remain_correct_for_parallel_sized_inputs() {
        let values = vec![2_i32; super::dispatch::PARALLEL_REDUCTION_THRESHOLD];
        let array =
            NDArray::from_shape_vec([super::dispatch::PARALLEL_REDUCTION_THRESHOLD], values)
                .unwrap();

        assert_eq!(
            array.sum().unwrap(),
            (super::dispatch::PARALLEL_REDUCTION_THRESHOLD as i32) * 2
        );
        assert_eq!(array.min().unwrap(), 2);
        assert_eq!(array.max().unwrap(), 2);
        assert_eq!(array.mean().unwrap(), 2.0);
    }

    #[test]
    fn axis_reductions_remain_correct_for_parallel_sized_inputs() {
        let rows = 512;
        let cols = super::dispatch::PARALLEL_REDUCTION_THRESHOLD / rows;
        let array = NDArray::from_shape_vec([rows, cols], vec![1.0_f64; rows * cols]).unwrap();

        let sum_axis_zero = array.sum_axis(0).unwrap();
        let mean_axis_one = array.mean_axis(1).unwrap();

        assert_eq!(sum_axis_zero.shape(), &[cols]);
        assert!(sum_axis_zero.data().iter().all(|value| *value == rows as f64));
        assert_eq!(mean_axis_one.shape(), &[rows]);
        assert!(mean_axis_one.data().iter().all(|value| *value == 1.0));
    }

    #[test]
    fn reduction_dispatch_stays_serial_for_small_and_medium_inputs() {
        assert!(!super::dispatch::should_parallelize_reduction_for_threads(1 << 18, 8));
        assert!(!super::dispatch::should_parallelize_reduction_for_threads((1 << 20) - 1, 8));
        assert!(!super::dispatch::should_parallelize_reduction_for_threads(1 << 20, 1));
    }

    #[test]
    fn reduction_dispatch_requires_enough_chunks_per_thread() {
        let work_items = super::dispatch::PARALLEL_REDUCTION_THRESHOLD;

        assert!(super::dispatch::should_parallelize_reduction_for_threads(work_items, 4));
        assert!(!super::dispatch::should_parallelize_reduction_for_threads(work_items, 33));
    }
}
