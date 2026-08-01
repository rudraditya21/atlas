use num_traits::ToPrimitive;

use super::{
    array::NDArray,
    axis::{AxisIndex, normalize_axis},
    error::{AtlasNdError, AtlasNdResult},
    stride::element_count,
    traits::Numeric,
    traversal::{for_each_value, offset_iter, try_for_each_value},
    view::ArrayView,
};

impl<T: Numeric> NDArray<T> {
    pub fn sum(&self) -> T {
        if self.is_contiguous() {
            return sum_contiguous(&self.data);
        }

        sum_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        if self.is_contiguous() {
            return prod_contiguous(&self.data);
        }

        prod_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return min_contiguous(&self.data, "min");
        }

        min_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return max_contiguous(&self.data, "max");
        }

        max_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        if self.is_contiguous() {
            return mean_contiguous(&self.data, "mean");
        }

        mean_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn sum_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        sum_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn prod_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        prod_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn min_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        min_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn max_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        max_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn mean_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn sum(&self) -> T {
        if self.is_contiguous() {
            return sum_contiguous(self.contiguous_slice());
        }

        sum_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        if self.is_contiguous() {
            return prod_contiguous(self.contiguous_slice());
        }

        prod_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return min_contiguous(self.contiguous_slice(), "min");
        }

        min_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return max_contiguous(self.contiguous_slice(), "max");
        }

        max_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        if self.is_contiguous() {
            return mean_contiguous(self.contiguous_slice(), "mean");
        }

        mean_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn sum_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        sum_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn prod_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        prod_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn min_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        min_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn max_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        max_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn mean_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    fn contiguous_slice(&self) -> &[T] {
        debug_assert!(self.is_contiguous());
        let len = element_count(&self.shape);
        &self.data[self.offset..self.offset + len]
    }
}

fn sum_contiguous<T: Numeric>(values: &[T]) -> T {
    let mut total = T::zero();

    for &value in values {
        total += value;
    }

    total
}

fn prod_contiguous<T: Numeric>(values: &[T]) -> T {
    let mut total = T::one();

    for &value in values {
        total *= value;
    }

    total
}

fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut minimum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value < minimum {
            minimum = value;
        }
    }

    Ok(minimum)
}

fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut maximum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value > maximum {
            maximum = value;
        }
    }

    Ok(maximum)
}

fn mean_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    if values.is_empty() {
        return Err(AtlasNdError::EmptyReduction { op });
    }

    let mut total = 0.0_f64;

    for &value in values {
        total += value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
    }

    Ok(total / values.len() as f64)
}

fn sum_axis_impl<T: Numeric>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    let mut reduced = Vec::with_capacity(element_count(&metadata.output_shape));

    for lane_offset in offset_iter(base_offset, &metadata.outer_shape, &metadata.outer_strides) {
        let mut total = T::zero();

        for step in 0..metadata.axis_len {
            total += data[lane_offset + step * metadata.axis_stride];
        }

        reduced.push(total);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn prod_axis_impl<T: Numeric>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    let mut reduced = Vec::with_capacity(element_count(&metadata.output_shape));

    for lane_offset in offset_iter(base_offset, &metadata.outer_shape, &metadata.outer_strides) {
        let mut total = T::one();

        for step in 0..metadata.axis_len {
            total *= data[lane_offset + step * metadata.axis_stride];
        }

        reduced.push(total);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn min_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    if metadata.axis_len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "min" });
    }

    let mut reduced = Vec::with_capacity(element_count(&metadata.output_shape));

    for lane_offset in offset_iter(base_offset, &metadata.outer_shape, &metadata.outer_strides) {
        let mut current = data[lane_offset];

        for step in 1..metadata.axis_len {
            let value = data[lane_offset + step * metadata.axis_stride];
            if value < current {
                current = value;
            }
        }

        reduced.push(current);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn max_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    if metadata.axis_len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "max" });
    }

    let mut reduced = Vec::with_capacity(element_count(&metadata.output_shape));

    for lane_offset in offset_iter(base_offset, &metadata.outer_shape, &metadata.outer_strides) {
        let mut current = data[lane_offset];

        for step in 1..metadata.axis_len {
            let value = data[lane_offset + step * metadata.axis_stride];
            if value > current {
                current = value;
            }
        }

        reduced.push(current);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    if metadata.axis_len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "mean" });
    }

    let mut reduced = Vec::with_capacity(element_count(&metadata.output_shape));

    for lane_offset in offset_iter(base_offset, &metadata.outer_shape, &metadata.outer_strides) {
        let mut total = 0.0_f64;

        for step in 0..metadata.axis_len {
            total += data[lane_offset + step * metadata.axis_stride]
                .to_f64()
                .ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
        }

        reduced.push(total / metadata.axis_len as f64);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

struct AxisReductionMetadata {
    output_shape: Vec<usize>,
    outer_shape: Vec<usize>,
    outer_strides: Vec<usize>,
    axis_stride: usize,
    axis_len: usize,
}

fn axis_reduction_metadata(
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<AxisReductionMetadata> {
    let axis = normalize_axis(axis, shape.len())?;

    let mut output_shape = Vec::with_capacity(shape.len().saturating_sub(1));
    let mut outer_strides = Vec::with_capacity(strides.len().saturating_sub(1));

    for (current_axis, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
        if current_axis == axis {
            continue;
        }

        output_shape.push(dim);
        outer_strides.push(stride);
    }

    Ok(AxisReductionMetadata {
        outer_shape: output_shape.clone(),
        output_shape,
        outer_strides,
        axis_stride: strides[axis],
        axis_len: shape[axis],
    })
}

fn sum_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::zero();
    for_each_value(data, offset, shape, strides, |value| {
        total += *value;
    });
    total
}

fn prod_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::one();
    for_each_value(data, offset, shape, strides, |value| {
        total *= *value;
    });
    total
}

fn min_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut minimum = None;

    for_each_value(data, offset, shape, strides, |value| {
        minimum = Some(match minimum {
            Some(current) if current < *value => current,
            Some(_) | None => *value,
        });
    });

    minimum.ok_or(AtlasNdError::EmptyReduction { op: "min" })
}

fn max_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut maximum = None;

    for_each_value(data, offset, shape, strides, |value| {
        maximum = Some(match maximum {
            Some(current) if current > *value => current,
            Some(_) | None => *value,
        });
    });

    maximum.ok_or(AtlasNdError::EmptyReduction { op: "max" })
}

fn mean_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let len = element_count(shape);
    if len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "mean" });
    }

    let mut total = 0.0_f64;
    try_for_each_value(data, offset, shape, strides, |value| {
        total += value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
        Ok(())
    })?;

    Ok(total / len as f64)
}
#[cfg(test)]
mod tests {
    use crate::{array::NDArray, error::AtlasNdError};

    #[test]
    fn whole_array_reductions_work_for_contiguous_arrays() {
        let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum(), 10);
        assert_eq!(array.prod(), 24);
        assert_eq!(array.min().unwrap(), 1);
        assert_eq!(array.max().unwrap(), 4);
        assert_eq!(array.mean().unwrap(), 2.5);
    }

    #[test]
    fn whole_array_reductions_work_for_scalar_arrays() {
        let array = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(array.sum(), 7);
        assert_eq!(array.prod(), 7);
        assert_eq!(array.min().unwrap(), 7);
        assert_eq!(array.max().unwrap(), 7);
        assert_eq!(array.mean().unwrap(), 7.0);
    }

    #[test]
    fn whole_array_reductions_work_for_non_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([0, 1], vec![2, 2]).unwrap();

        assert_eq!(view.sum(), 12);
        assert_eq!(view.prod(), 40);
        assert_eq!(view.min().unwrap(), 1);
        assert_eq!(view.max().unwrap(), 5);
        assert_eq!(view.mean().unwrap(), 3.0);
    }

    #[test]
    fn sum_and_prod_use_identity_for_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7);

        assert_eq!(array.sum(), 0);
        assert_eq!(array.prod(), 1);
    }

    #[test]
    fn min_max_and_mean_reject_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7);

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
    fn axis_reductions_validate_axis_bounds() {
        let array = NDArray::new(vec![2, 3], 1_i32);

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
        let array = NDArray::<i32>::new(vec![0, 3], 1);

        assert_eq!(array.sum_axis(0).unwrap().shape(), &[3]);
        assert_eq!(array.sum_axis(0).unwrap().data(), &[0, 0, 0]);
        assert_eq!(array.prod_axis(0).unwrap().data(), &[1, 1, 1]);
        assert_eq!(array.min_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(array.max_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(array.mean_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn axis_reductions_preserve_zero_length_output_shapes_when_lanes_are_empty() {
        let array = NDArray::<i32>::new(vec![2, 0, 3], 1);

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
    fn axis_reductions_return_scalar_outputs_for_one_dimensional_inputs() {
        let array = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum_axis(-1).unwrap().shape(), &[] as &[usize]);
        assert_eq!(array.sum_axis(-1).unwrap().data(), &[10]);
        assert_eq!(array.prod_axis(-1).unwrap().data(), &[24]);
        assert_eq!(array.min_axis(-1).unwrap().data(), &[1]);
        assert_eq!(array.max_axis(-1).unwrap().data(), &[4]);
        assert_eq!(array.mean_axis(-1).unwrap().data(), &[2.5]);
    }
}
